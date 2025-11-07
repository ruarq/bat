mod audio;

use audio::AnalysisData;
use cpal::traits::StreamTrait;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

fn main() -> eframe::Result {
    std::thread::sleep(std::time::Duration::from_secs(1));

    let (sender, receiver) = crossbeam_channel::bounded(1);
    let (producer, consumer) = rtrb::RingBuffer::new(audio::RING_BUFFER_CAPACITY);
    let audio_buffer_size_selected = AudioBufferSize::default();
    let audio_buffer_size = Arc::new(AtomicUsize::new(audio_buffer_size_selected as usize));
    let audio_buffer_size2 = audio_buffer_size.clone();
    let analyze_audio_thread =
        thread::spawn(|| audio::analyze_audio(consumer, sender, audio_buffer_size2));

    let (stream, stream_config) = audio::build_audio_input_stream(producer);
    stream.play().expect("unable to start stream");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1240.0, 720.0]),
        ..Default::default()
    };
    let eframe_result = eframe::run_native(
        "bat",
        options,
        Box::new(|_| {
            Ok(Box::<App>::new({
                App {
                    stream_config,
                    receiver,
                    spectrum_slope: 4.5,
                    analysis_data: Default::default(),
                    audio_buffer_size,
                    audio_buffer_size_selected,
                }
            }))
        }),
    );

    analyze_audio_thread
        .join()
        .expect("failed to join analyze_audio_thread");

    return eframe_result;
}

struct App {
    stream_config: cpal::StreamConfig,
    receiver: crossbeam_channel::Receiver<AnalysisData>,
    spectrum_slope: f32,
    analysis_data: AnalysisData,
    audio_buffer_size: Arc<AtomicUsize>,
    audio_buffer_size_selected: AudioBufferSize,
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(usize)]
enum AudioBufferSize {
    Small = 1 << 10,
    Medium = 1 << 11,
    Big = 1 << 12,
    Huge = 1 << 13,
}

impl Default for AudioBufferSize {
    fn default() -> Self {
        Self::Medium
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.receiver.try_recv() {
                Ok(data) => self.analysis_data = data,
                Err(_) => {}
            };

            let audio_buffer_size = self.audio_buffer_size.load(Ordering::Relaxed);

            let (left, right) = self.analysis_data.rms_meter;
            let meter_range = (-96.0, 0.0);

            let meter_left = audio::make_meter(audio::as_decibel(left), meter_range);
            let meter_right = audio::make_meter(audio::as_decibel(right), meter_range);

            let mut spectrum = self.analysis_data.spectrum.clone();
            audio::smooth_linear(5, &mut spectrum);
            audio::tilt(self.spectrum_slope, &mut spectrum);

            ui.add(egui::ProgressBar::new(meter_left));
            ui.add(egui::ProgressBar::new(meter_right));
            ui.add(egui::Slider::new(&mut self.spectrum_slope, -10.0..=10.0));

            let before = self.audio_buffer_size_selected;
            egui::ComboBox::from_label("buffer size")
                .selected_text(format!("{:?}", self.audio_buffer_size_selected))
                .show_ui(ui, |ui| {
                    use AudioBufferSize::*;
                    let selectable = [Small, Medium, Big, Huge];
                    for s in selectable {
                        ui.selectable_value(
                            &mut self.audio_buffer_size_selected,
                            s,
                            format!("{:?}", s),
                        );
                    }
                });

            if before != self.audio_buffer_size_selected {
                self.audio_buffer_size
                    .store(self.audio_buffer_size_selected as usize, Ordering::Relaxed);
            }

            egui_plot::Plot::new("Frequency Spectrum")
                .legend(egui_plot::Legend::default())
                .clamp_grid(false)
                //.x_grid_spacer(egui_plot::log_grid_spacer(10))
                .x_axis_label("Hz")
                .x_grid_spacer(egui_plot::log_grid_spacer(10))
                .y_axis_label("dB")
                .allow_drag(false)
                .allow_scroll(false)
                .default_x_bounds(1.0, 5.0)
                .default_y_bounds(meter_range.0 as f64, meter_range.1 as f64)
                .show(ui, |plot_ui| {
                    plot_ui.line(egui_plot::Line::new(
                        "FFT",
                        audio::frequencies(
                            audio_buffer_size as u32 / self.stream_config.channels as u32,
                            self.stream_config.sample_rate.0,
                        )
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

            ctx.request_repaint()
        });
    }
}
