mod audio;

use audio::{AnalysisData, AudioBufferSize};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};
use strum::IntoEnumIterator;

fn main() -> eframe::Result {
    std::thread::sleep(std::time::Duration::from_secs(1));

    let (sender, receiver) = crossbeam_channel::bounded(audio::CROSSBEAM_CHANNEL_CAPACITY);
    let (producer, consumer) = rtrb::RingBuffer::new(audio::RINGBUFFER_CAPACITY);
    let audio_buffer_size_selected = AudioBufferSize::default();
    let audio_buffer_size = Arc::new(AtomicUsize::new(audio_buffer_size_selected as usize));
    let audio_buffer_size2 = audio_buffer_size.clone();
    let analysis_thread =
        thread::spawn(|| audio::analyze_audio(consumer, sender, audio_buffer_size2));

    let sample_rate = audio::SampleRate::Audio2;
    let (host, stream, stream_config) = audio::build_audio_input_stream(0, sample_rate, producer);
    stream.play().unwrap();

    let app = App {
        host,
        stream,
        stream_config,
        stream_playing: true,
        analysis_thread,
        audio_device_index: 0,
        receiver,
        spectrum_slope: 4.5,
        spectrum_range: (20.0, 18_000.0),
        spectrum_smoothing: 5,
        analysis_data: Default::default(),
        audio_buffer_size,
        audio_buffer_size_selected,
        sample_rate,
        meter_range: (-96.0, 0.0),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1920.0, 1080.0]),
        ..Default::default()
    };

    eframe::run_native("bat", options, Box::new(|_| Ok(Box::new(app))))
}

struct App {
    host: cpal::Host,
    stream: cpal::Stream,
    stream_config: cpal::StreamConfig,
    stream_playing: bool,
    analysis_thread: JoinHandle<()>,
    audio_device_index: usize,
    receiver: crossbeam_channel::Receiver<AnalysisData>,
    spectrum_slope: f32,
    spectrum_range: (f32, f32),
    spectrum_smoothing: usize,
    analysis_data: AnalysisData,
    audio_buffer_size: Arc<AtomicUsize>,
    audio_buffer_size_selected: audio::AudioBufferSize,
    sample_rate: audio::SampleRate,
    meter_range: (f32, f32),
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.receiver.try_recv() {
                Ok(data) => self.analysis_data = data,
                Err(e) => eprintln!("ui thread: {}", e),
            };

            self.draw_controls(ui);
            self.draw_rms_meter(ui);
            self.draw_spectrum_plot(ui);

            ctx.request_repaint()
        });
    }
}

impl App {
    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.spectrum_slope, 0.0..=6.0));
            ui.add(egui::Slider::new(&mut self.spectrum_smoothing, 1..=1000).logarithmic(true));
        });

        if ui
            .add(egui::Button::new(if self.stream_playing {
                "playing"
            } else {
                "paused"
            }))
            .clicked()
        {
            self.stream_playing = !self.stream_playing;
            if self.stream_playing {
                self.stream.play().unwrap();
            } else {
                self.stream.pause().unwrap();
            }
        }

        let audio_buffer_size_before = self.audio_buffer_size_selected;
        egui::ComboBox::from_label("audio buffer size")
            .selected_text(format!("{}", self.audio_buffer_size_selected as usize))
            .show_ui(ui, |ui| {
                for s in AudioBufferSize::iter() {
                    ui.selectable_value(
                        &mut self.audio_buffer_size_selected,
                        s,
                        format!("{} ({})", s, s as usize),
                    );
                }
            });
        if audio_buffer_size_before != self.audio_buffer_size_selected {
            self.audio_buffer_size
                .store(self.audio_buffer_size_selected as usize, Ordering::Relaxed);
        }

        let format_sample_rate =
            |sample_rate| format!("{:.1} kHz", (sample_rate as u32) as f32 / 1000.0);
        let sample_rate_before = self.sample_rate;
        egui::ComboBox::from_label("sample rate")
            .selected_text(format_sample_rate(self.sample_rate))
            .show_ui(ui, |ui| {
                for s in audio::SampleRate::iter() {
                    ui.selectable_value(&mut self.sample_rate, s, format_sample_rate(s));
                }
            });
        if sample_rate_before != self.sample_rate {
            self.reinit_audio_input_stream();
        }

        let audio_device_index_before = self.audio_device_index;
        egui::ComboBox::from_label("audio input device")
            .selected_text(format!(
                "{}",
                self.host
                    .input_devices()
                    .unwrap()
                    .nth(self.audio_device_index)
                    .unwrap()
                    .name()
                    .unwrap()
            ))
            .show_ui(ui, |ui| {
                for (i, d) in self.host.input_devices().unwrap().enumerate() {
                    ui.selectable_value(
                        &mut self.audio_device_index,
                        i,
                        format!("{}", d.name().unwrap()),
                    );
                }
            });
        if audio_device_index_before != self.audio_device_index {
            self.reinit_audio_input_stream();
        }
    }

    fn reinit_audio_input_stream(&mut self) {
        let (producer, consumer) = rtrb::RingBuffer::new(audio::RINGBUFFER_CAPACITY);
        let (sender, receiver) = crossbeam_channel::bounded(audio::CROSSBEAM_CHANNEL_CAPACITY);
        let audio_buffer_size = self.audio_buffer_size.clone();

        (self.host, self.stream, self.stream_config) =
            audio::build_audio_input_stream(self.audio_device_index, self.sample_rate, producer);
        self.stream.play().unwrap();
        self.analysis_thread =
            thread::spawn(|| audio::analyze_audio(consumer, sender, audio_buffer_size));

        self.receiver = receiver;
    }

    fn draw_rms_meter(&mut self, ui: &mut egui::Ui) {
        let (left, right) = self.analysis_data.rms_meter;
        let meter_left = audio::make_meter(audio::as_decibel(left), self.meter_range);
        let meter_right = audio::make_meter(audio::as_decibel(right), self.meter_range);

        ui.vertical(|ui| {
            for meter in [meter_left, meter_right] {
                ui.add(
                    egui::ProgressBar::new(meter), /* .desired_width(50.0) */
                );
            }
        });
    }

    fn spectrum_grid_stage(step_size: f64) -> Vec<egui_plot::GridMark> {
        (1..=10)
            .map(|i| egui_plot::GridMark {
                value: (step_size * i as f64).log10(),
                step_size: step_size.log10(),
            })
            .collect()
    }

    fn spectrum_grid(_: egui_plot::GridInput) -> Vec<egui_plot::GridMark> {
        let mut marks = Vec::new();

        marks.append(&mut Self::spectrum_grid_stage(10.0));
        marks.append(&mut Self::spectrum_grid_stage(100.0));
        marks.append(&mut Self::spectrum_grid_stage(1000.0));

        marks
    }

    fn draw_spectrum_plot(&mut self, ui: &mut egui::Ui) {
        let audio_buffer_size = self.audio_buffer_size.load(Ordering::Relaxed);
        let fft_buffer_size = audio_buffer_size / self.stream_config.channels as usize;

        let mut spectrum = self.analysis_data.spectrum.clone();

        audio::smooth_linear(self.spectrum_smoothing as usize, &mut spectrum);
        audio::tilt(self.spectrum_slope, &mut spectrum);

        let format_frequency = |f: f64| {
            let f = f.round();
            if f < 1000.0 {
                format!("{}", f)
            } else {
                format!("{}k", f / 1000.0)
            }
        };

        egui_plot::Plot::new("Frequency Spectrum")
            .legend(egui_plot::Legend::default())
            .clamp_grid(false)
            .x_axis_label("Hz")
            .x_axis_formatter(|grid_mark, _| format_frequency(10f64.powf(grid_mark.value)))
            .x_grid_spacer(Self::spectrum_grid)
            .y_axis_label("dB")
            .allow_drag(false)
            .allow_scroll(false)
            .default_x_bounds(
                self.spectrum_range.0.log10() as f64,
                self.spectrum_range.1.log10() as f64,
            )
            .default_y_bounds(self.meter_range.0 as f64, self.meter_range.1 as f64)
            //.min_size(egui::Vec2::new(200.0, 500.0))
            .show(ui, |plot_ui| {
                plot_ui.line(egui_plot::Line::new(
                    format!("FFT {}", fft_buffer_size),
                    audio::frequencies(fft_buffer_size as u32, self.stream_config.sample_rate.0)
                        .iter()
                        .zip(spectrum)
                        .skip(1)
                        .map(|(freq, amp)| {
                            [
                                freq.log10() as f64,
                                audio::as_decibel(amp).max(-1e10) as f64,
                            ]
                        })
                        .collect::<egui_plot::PlotPoints<'_>>(),
                ))
            });
    }
}
