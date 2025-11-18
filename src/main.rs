mod audio;
mod discovery;
mod gradient;
mod led;

use audio::{AnalysisData, AudioBufferSize};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use discovery::{Discovery, DiscoveryConfig};
use led::LedStrip;
use std::{
    net::{SocketAddr, UdpSocket},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};
use strum::{Display, EnumIter, IntoEnumIterator};

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

    let (client_queue_sender, client_queue_receiver) = crossbeam_channel::unbounded();

    let mut discovery = Discovery::new(
        DiscoveryConfig::default(),
        move |socket, message, sender| {
            if message == "beatstrip client discovery request".as_bytes() {
                let message = "beatstrip discovery ok".as_bytes();
                let bytes_sent = socket.send_to(message, sender).unwrap_or(0);
                if bytes_sent == message.len() {
                    client_queue_sender.send(sender).unwrap();
                }
            }
        },
    );

    discovery.start().unwrap();

    let app = App {
        panel: Default::default(),
        host,
        stream,
        stream_config,
        stream_playing: true,
        analysis_thread,
        audio_device_index: 0,
        receiver,
        spectrum_slope: 4.5,
        spectrum_range: (20.0, 18_000.0),
        spectrum_smoothing: Default::default(),
        spectrum_smoothing_lin: 5,
        spectrum_smoothing_log: 24.0,
        analysis_data: Default::default(),
        audio_buffer_size,
        audio_buffer_size_selected,
        sample_rate,
        meter_range: (-96.0, 12.0),
        led_strips: vec![
            LedStrip::with_size(60),
            LedStrip::with_size(60),
            LedStrip::with_size(60),
            LedStrip::with_size(60),
        ],
        master_gain: 1.0,
        client_queue_receiver,
        clients: Vec::new(),
        socket: UdpSocket::bind("0.0.0.0:25120").unwrap(),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1920.0, 1080.0]),
        ..Default::default()
    };

    eframe::run_native("bat", options, Box::new(|_| Ok(Box::new(app))))
}

#[derive(Debug, EnumIter, Display, PartialEq, Clone, Copy)]
enum Panel {
    AudioVisualization,
    Leds,
}

impl Default for Panel {
    fn default() -> Self {
        Self::AudioVisualization
    }
}

#[derive(Debug, EnumIter, Display, PartialEq, Clone, Copy)]
enum SpectrumSmoothing {
    None,
    Linear,
    Logarithmic,
}

impl Default for SpectrumSmoothing {
    fn default() -> Self {
        Self::None
    }
}

struct App {
    panel: Panel,
    host: cpal::Host,
    stream: cpal::Stream,
    stream_config: cpal::StreamConfig,
    stream_playing: bool,
    analysis_thread: JoinHandle<()>,
    audio_device_index: usize,
    receiver: crossbeam_channel::Receiver<AnalysisData>,
    spectrum_slope: f32,
    spectrum_range: (f32, f32),
    spectrum_smoothing: SpectrumSmoothing,
    spectrum_smoothing_lin: usize,
    spectrum_smoothing_log: f32,
    analysis_data: AnalysisData,
    audio_buffer_size: Arc<AtomicUsize>,
    audio_buffer_size_selected: audio::AudioBufferSize,
    sample_rate: audio::SampleRate,
    meter_range: (f32, f32),
    led_strips: Vec<LedStrip>,
    master_gain: f32,
    client_queue_receiver: crossbeam_channel::Receiver<SocketAddr>,
    clients: Vec<SocketAddr>,
    socket: UdpSocket,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        match self.receiver.try_recv() {
            Ok(mut data) => {
                data.rms_meter.0 *= self.master_gain;
                data.rms_meter.1 *= self.master_gain;
                data.spectrum
                    .iter_mut()
                    .for_each(|s| *s *= *s * self.master_gain);
                self.analysis_data = data;
            }
            Err(e) => eprintln!("ui thread: {}", e),
        };

        match self.client_queue_receiver.try_recv() {
            Ok(addr) => self.clients.push(addr),
            Err(_) => {}
        }

        for client in self.clients.iter() {
            let mut strip = LedStrip::with_size(60);

            let mut spectrum = self.analysis_data.spectrum.clone();
            spectrum
                .iter_mut()
                .for_each(|s| *s = audio::make_meter(audio::as_decibel(*s), self.meter_range));
            use SpectrumSmoothing::*;
            match self.spectrum_smoothing {
                None => {}
                Linear => audio::smooth_lin(self.spectrum_smoothing_lin, &mut spectrum),
                Logarithmic => {
                    (spectrum, _) = audio::smooth_log(
                        self.spectrum_smoothing_log,
                        self.sample_rate as u32,
                        &spectrum,
                    );
                }
            }
            audio::tilt(self.spectrum_slope, &mut spectrum);

            strip.biamps(&spectrum[..strip.data.len()]);

            self.socket
                .send_to(bytemuck::cast_slice(&strip.data), client)
                .unwrap();
        }

        egui::TopBottomPanel::top("Panels").show(ctx, |ui| {
            self.draw_panel_selection(ui);
        });

        egui::SidePanel::right("Settings").show(ctx, |ui| {
            self.draw_controls(ui);
        });

        match self.panel {
            Panel::AudioVisualization => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    self.draw_rms_meter(ui);
                    self.draw_spectrum_plot(ui);
                });
            }

            Panel::Leds => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let (w, h) = (10.0, 10.0);
                    for strip_num in 0..5 {
                        for strip in &mut self.led_strips.iter_mut() {
                            let rms = (self.analysis_data.rms_meter.0
                                + self.analysis_data.rms_meter.1)
                                / 2.0;
                            let meter = audio::make_meter(audio::as_decibel(rms), self.meter_range);
                            let mut spectrum = self.analysis_data.spectrum.clone();
                            spectrum.iter_mut().for_each(|s| {
                                *s = audio::make_meter(audio::as_decibel(*s), self.meter_range)
                            });

                            use SpectrumSmoothing::*;
                            match self.spectrum_smoothing {
                                None => {}
                                Linear => {
                                    audio::smooth_lin(self.spectrum_smoothing_lin, &mut spectrum)
                                }
                                Logarithmic => {
                                    (spectrum, _) = audio::smooth_log(
                                        self.spectrum_smoothing_log,
                                        self.sample_rate as u32,
                                        &spectrum,
                                    );
                                }
                            }

                            audio::tilt(self.spectrum_slope, &mut spectrum);

                            match strip_num % 5 {
                                0 => strip.amp(meter),
                                1 => strip.amp(1.0 - meter),
                                2 => strip.biamp(meter),
                                3 => strip.amps(&spectrum[..strip.data.len()]),
                                4 => strip.biamps(&spectrum[..strip.data.len()]),
                                _ => panic!("impossible"),
                            }

                            let (response, painter) = ui.allocate_painter(
                                egui::vec2(strip.data.len() as f32 * (w + 1.0), h),
                                egui::Sense::hover(),
                            );

                            for (i, rgb) in strip.data.iter().enumerate() {
                                let (x, y) = (w * i as f32, 0.0);
                                painter.rect_filled(
                                    egui::Rect {
                                        min: response.rect.min + egui::vec2(x, y),
                                        max: response.rect.min + egui::vec2(x + w, y + h),
                                    },
                                    egui::CornerRadius::ZERO,
                                    egui::Color32::from_rgb(rgb.r(), rgb.g(), rgb.b()),
                                );
                            }
                        }
                    }
                });
            }
        }

        egui::Window::new("Statistics").show(ctx, |ui| {
            ui.heading("Performance");
            let mut frametime_ms = 0.0;
            ui.input(|input| {
                frametime_ms = input.stable_dt * 1000.0;
            });
            ui.label(format!("Frametime: {:.3}ms", frametime_ms));

            ui.label(format!(
                "Analysis thread: {:.3}ms",
                self.analysis_data.elapsed.as_millis()
            ));

            ui.separator();

            let sample_rate = self.sample_rate as usize;
            let buffer_size = self.audio_buffer_size_selected as usize;
            let fft_buffer_size = buffer_size / self.stream_config.channels as usize;

            ui.heading("Analysis info");
            ui.label(format!(
                "Time resolution: {:.2}ms",
                (buffer_size as f32 / sample_rate as f32) * 1000.0
            ));

            ui.label(format!(
                "Frequency resolution: {:.2} Hz/bin",
                audio::frequency_resolution(fft_buffer_size as u32, sample_rate as u32)
            ));
        });

        ctx.request_repaint()
    }
}

impl App {
    fn draw_panel_selection(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for p in Panel::iter() {
                ui.selectable_value(&mut self.panel, p, p.to_string());
            }
        });
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Spectrum");

        egui::ComboBox::from_label("Smoothing")
            .selected_text(self.spectrum_smoothing.to_string())
            .show_ui(ui, |ui| {
                for smoothing in SpectrumSmoothing::iter() {
                    ui.selectable_value(
                        &mut self.spectrum_smoothing,
                        smoothing,
                        smoothing.to_string(),
                    );
                }
            });

        ui.vertical(|ui| {
            use SpectrumSmoothing::*;
            let smoothing_slider_label = "smooth amount";
            match self.spectrum_smoothing {
                None => (),
                Linear => {
                    ui.add(
                        egui::Slider::new(&mut self.spectrum_smoothing_lin, 1..=1000)
                            .logarithmic(true)
                            .text(smoothing_slider_label),
                    );
                }
                Logarithmic => {
                    ui.add(
                        egui::Slider::new(&mut self.spectrum_smoothing_log, 3.0..=48.0)
                            .text(smoothing_slider_label),
                    );
                }
            }

            ui.add(egui::Slider::new(&mut self.spectrum_slope, 0.0..=6.0).text("Slope"));
        });

        ui.separator();

        ui.heading("Audio stream");

        ui.label("Master Gain");
        ui.add(egui::Slider::new(&mut self.master_gain, 0.0..=2.0));

        ui.label("Meter range");
        ui.group(|ui| {
            ui.add(egui::Slider::new(
                &mut self.meter_range.0,
                -96.0..=(self.meter_range.1 - 0.1),
            ));
            ui.add(egui::Slider::new(
                &mut self.meter_range.1,
                self.meter_range.0..=6.0,
            ));
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
        let mut spectrum = self.analysis_data.spectrum.clone();
        let fft_buffer_size = self.audio_buffer_size_selected as usize / 2;
        let mut frequencies = audio::frequencies(spectrum.len() as u32, self.sample_rate as u32);

        use SpectrumSmoothing::*;
        match self.spectrum_smoothing {
            None => {}
            Linear => audio::smooth_lin(self.spectrum_smoothing_lin, &mut spectrum),
            Logarithmic => {
                (spectrum, frequencies) = audio::smooth_log(
                    self.spectrum_smoothing_log,
                    self.sample_rate as u32,
                    &spectrum,
                );
            }
        }

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
                    frequencies
                        .iter()
                        .zip(spectrum)
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
