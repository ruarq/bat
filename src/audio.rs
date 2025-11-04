use cpal::traits::{DeviceTrait, HostTrait};
use realfft::{RealFftPlanner, num_complex::Complex};
use std::{thread, time::Duration};

pub const RING_BUFFER_CAPACITY: usize = 8192;
pub const AUDIO_BUFFER_SIZE: usize = 2048 * 2;

pub struct AnalysisData {
    pub rms: (f32, f32),
    pub spectrum: Vec<Complex<f32>>,
}

impl Default for AnalysisData {
    fn default() -> Self {
        Self {
            rms: (0.0, 0.0),
            spectrum: Vec::new(),
        }
    }
}

pub fn analyze_audio(
    mut audio_consumer: rtrb::Consumer<f32>,
    ui_sender: crossbeam_channel::Sender<AnalysisData>,
) {
    let buffer_size = AUDIO_BUFFER_SIZE;
    let mut audio_buffer = vec![0.0f32; buffer_size];

    let mut fft_planner = RealFftPlanner::new();
    let fft = fft_planner.plan_fft_forward(buffer_size / 2);
    let mut spectrum = fft.make_output_vec();

    loop {
        match audio_consumer.read_chunk(buffer_size) {
            Ok(chunk) => {
                for (dest, src) in audio_buffer.iter_mut().zip(chunk.into_iter()) {
                    *dest = src;
                }

                let left = audio_buffer.iter().step_by(2);
                let mut right = audio_buffer.iter();
                right.next();
                let right = right.step_by(2);

                let left_rms = left.clone().map(|s| s * s).sum::<f32>() / left.len() as f32;
                let right_rms = right.clone().map(|s| s * s).sum::<f32>() / right.len() as f32;

                let mut mid: Vec<f32> = left
                    .clone()
                    .zip(right.clone())
                    .map(|(l, r)| (l + r) / 2.0)
                    .collect();

                fft.process(&mut mid, &mut spectrum)
                    .expect("fft.process(...): something went wrong");

                spectrum
                    .iter_mut()
                    .for_each(|c| *c = *c / (mid.len() as f32).sqrt());

                match ui_sender.send(AnalysisData {
                    rms: (left_rms, right_rms),
                    spectrum: spectrum.clone(),
                }) {
                    Ok(()) => {}
                    Err(e) =>
                        /* eprintln!("failed to send analysis data: {}", e) */
                        {}
                }
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(5));
                continue;
            }
        }
    }
}

pub fn build_audio_input_stream(mut producer: rtrb::Producer<f32>) -> cpal::Stream {
    let host = cpal::default_host();
    host.input_devices()
        .expect("no input devices available")
        .enumerate()
        .for_each(|(i, d)| {
            println!(
                "{} - {}",
                i,
                d.name().unwrap_or(String::from("NAME_UNKNOWN"))
            )
        });

    print!(">>> ");
    let mut device_index = String::new();
    std::io::stdin()
        .read_line(&mut device_index)
        .expect("failed to read device index");
    let device_index: usize = device_index
        .trim()
        .parse()
        .expect("failed to parse device_index");

    let device = host
        .input_devices()
        .unwrap()
        .nth(device_index)
        .expect("invalid device_index");

    eprintln!(
        "using device: {}",
        device
            .name()
            .unwrap_or(String::from("couldn't obtain name"))
    );

    let config = device
        .default_input_config()
        .expect("no device input config");
    let format = config.sample_format();
    let channels = config.channels();
    let config = config.into();

    eprintln!("using {}@{}ch", format, channels);

    let data_callback = move |data: &[f32], _info: &cpal::InputCallbackInfo| match producer
        .write_chunk_uninit(data.len())
    {
        Ok(chunk) => {
            chunk.fill_from_iter(data.iter().copied());
        }
        Err(e) => { /* eprintln!("error pushing audio data: {}", e); */ }
    };

    let err_fn = |err| eprintln!("input stream error: {}", err);

    match format {
        cpal::SampleFormat::F32 => device.build_input_stream(&config, data_callback, err_fn, None),
        format => panic!("unsupported format: {}", format),
    }
    .expect("couldn't build input stream")
}

pub fn as_decibel(rms: f32) -> f32 {
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        f32::NEG_INFINITY
    }
}

pub fn make_meter(value: f32, range: (f32, f32)) -> f32 {
    let (lower, upper) = range;
    ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
}
