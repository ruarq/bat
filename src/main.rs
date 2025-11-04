mod audio;

use audio::AnalysisData;
use cpal::traits::StreamTrait;
use std::thread;

fn main() -> eframe::Result {
    std::thread::sleep(std::time::Duration::from_secs(1));

    let (sender, receiver) = crossbeam_channel::bounded(1);
    let (producer, consumer) = rtrb::RingBuffer::new(audio::RING_BUFFER_CAPACITY);
    let analyze_audio_thread = thread::spawn(|| audio::analyze_audio(consumer, sender));

    let stream = audio::build_audio_input_stream(producer);
    stream.play().expect("unable to start stream");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1240.0, 720.0]),
        ..Default::default()
    };
    let eframe_result = eframe::run_native(
        "bat",
        options,
        Box::new(|_| Ok(Box::<App>::new(App { receiver }))),
    );

    analyze_audio_thread
        .join()
        .expect("failed to join analyze_audio_thread");

    return eframe_result;
}

struct App {
    receiver: crossbeam_channel::Receiver<AnalysisData>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data = match self.receiver.recv() {
                Ok(data) => data,
                Err(e) => {
                    /* eprintln!("failed to fetch analysis data: {}", e); */
                    Default::default()
                }
            };

            let (left, right) = data.rms;
            let meter_range = (-96.0, 0.0);

            let meter_left = audio::make_meter(audio::as_decibel(left), meter_range);
            let meter_right = audio::make_meter(audio::as_decibel(right), meter_range);

            ui.add(egui::ProgressBar::new(meter_left));
            ui.add(egui::ProgressBar::new(meter_right));

            egui_plot::Plot::new("Frequency Spectrum")
                .legend(egui_plot::Legend::default())
                .clamp_grid(false)
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(
                        egui_plot::BarChart::new(
                            "barchart",
                            data.spectrum
                                .iter()
                                .map(|c| c.re.abs())
                                .enumerate()
                                .map(|(x, r)| {
                                    egui_plot::Bar::new(
                                        (x as f64).log10().clamp(0.0, data.spectrum.len() as f64),
                                        audio::make_meter(audio::as_decibel(r), (-96.0, 0.0))
                                            as f64,
                                    )
                                })
                                .collect(),
                        )
                        .width(0.01),
                    )
                });

            ctx.request_repaint();
        });
    }
}
