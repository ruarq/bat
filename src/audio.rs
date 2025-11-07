use cpal::traits::{DeviceTrait, HostTrait};
use realfft::{RealFftPlanner, num_complex::Complex};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

pub const RING_BUFFER_CAPACITY: usize = 8192 * 2;
//pub const AUDIO_BUFFER_SIZE: usize = 2048 * 2;
//pub const FFT_BUFFER_SIZE: usize = AUDIO_BUFFER_SIZE / 2;

#[derive(Clone)]
pub struct AnalysisData {
    pub rms_meter: (f32, f32),

    /// normalized fft
    pub spectrum: Vec<f32>,
}

impl Default for AnalysisData {
    fn default() -> Self {
        Self {
            rms_meter: (0.0, 0.0),
            spectrum: Vec::new(),
        }
    }
}

pub fn analyze_audio(
    mut audio_consumer: rtrb::Consumer<f32>,
    ui_sender: crossbeam_channel::Sender<AnalysisData>,
    audio_buffer_size: Arc<AtomicUsize>,
) {
    let mut buffer_size = audio_buffer_size.load(Ordering::Relaxed);
    let mut fft_size = buffer_size / 2; // assuming 2ch audio
    let mut audio_buffer = vec![0.0f32; buffer_size];

    let mut fft_planner = RealFftPlanner::new();
    let mut fft = fft_planner.plan_fft_forward(fft_size);
    let mut spectrum = fft.make_output_vec();

    loop {
        let new_buffer_size = audio_buffer_size.load(Ordering::Relaxed);
        if new_buffer_size != buffer_size {
            buffer_size = new_buffer_size;
            fft_size = buffer_size / 2;
            audio_buffer.resize(buffer_size, 0.0);
            fft = fft_planner.plan_fft_forward(fft_size);
            spectrum = fft.make_output_vec();
        }

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

                //let samples = mid.iter().map(|s| *s).collect();

                fft.process(&mut mid, &mut spectrum)
                    .expect("fft.process(...): something went wrong");

                let normalize = |c: &mut Complex<f32>| *c = 2.0 * *c / (fft_size as f32);

                spectrum.iter_mut().for_each(normalize);

                match ui_sender.send(AnalysisData {
                    rms_meter: (left_rms, right_rms),
                    spectrum: spectrum.iter().map(|c| c.re.abs()).collect(),
                }) {
                    Ok(()) => {}
                    Err(_) =>
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

pub fn build_audio_input_stream(
    device_index: usize,
    mut producer: rtrb::Producer<f32>,
) -> (cpal::Host, cpal::Stream, cpal::StreamConfig) {
    let host = cpal::default_host();

    //host.input_devices()
    //    .expect("no input devices available")
    //    .enumerate()
    //    .for_each(|(i, d)| {
    //        println!(
    //            "{} - {}",
    //            i,
    //            d.name().unwrap_or(String::from("NAME_UNKNOWN"))
    //        )
    //    });

    //print!(">>> ");
    //let mut device_index = String::new();
    //std::io::stdin()
    //    .read_line(&mut device_index)
    //    .expect("failed to read device index");
    //let device_index: usize = device_index
    //    .trim()
    //    .parse()
    //  .expect("failed to parse device_index");

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
        Err(_) => { /* eprintln!("error pushing audio data: {}", e); */ }
    };

    let err_fn = |err| eprintln!("input stream error: {}", err);

    let stream = match format {
        cpal::SampleFormat::F32 => device.build_input_stream(&config, data_callback, err_fn, None),
        format => panic!("unsupported format: {}", format),
    }
    .expect("couldn't build input stream");

    (host, stream, config)
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

pub fn frequencies(fft_size: u32, sample_rate: u32) -> Vec<f32> {
    let interval = sample_rate as f32 / fft_size as f32;
    (0..fft_size).map(|k| k as f32 * interval).collect()
}

pub fn tilt(slope: f32, spectrum: &mut [f32]) {
    let alpha = slope / (20.0 * 2.0f32.log10());
    spectrum.iter_mut().enumerate().for_each(|(i, s)| {
        let gain = (i as f32).powf(alpha);
        *s *= gain;
    })
}

pub fn smooth_linear(window_size: usize, spectrum: &mut [f32]) {
    // TODO: can we eliminate this allocation?

    let smoothed_spectrum: Vec<f32> = spectrum
        .windows(window_size)
        .map(|win| win.iter().sum::<f32>() / win.len() as f32)
        .collect();

    spectrum
        .iter_mut()
        .zip(smoothed_spectrum)
        .for_each(|(s, t)| *s = t);
}

pub fn _smooth_logarithmic(_window_size: f32, _spectrum: &mut [f32]) {
    todo!()
}
