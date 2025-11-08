use cpal::traits::{DeviceTrait, HostTrait};
use realfft::{RealFftPlanner, num_complex::Complex};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use strum::{Display, EnumIter};

pub const RINGBUFFER_CAPACITY: usize = 8192 * 4;
pub const CROSSBEAM_CHANNEL_CAPACITY: usize = 1;

#[derive(Clone)]
pub struct AnalysisData {
    pub rms_meter: (f32, f32),

    /// normalized fft
    pub spectrum: Vec<f32>,

    pub elapsed: Duration,
}

impl Default for AnalysisData {
    fn default() -> Self {
        Self {
            rms_meter: (0.0, 0.0),
            spectrum: Vec::new(),
            elapsed: Duration::ZERO,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, EnumIter, Display)]
#[repr(usize)]
pub enum AudioBufferSize {
    Microscopic = 1 << 8,
    Mini = 1 << 9,
    Small = 1 << 10,
    Medium = 1 << 11,
    Big = 1 << 12,
    Huge = 1 << 13,
    Insane = 1 << 14,
    Otherworldy = 1 << 15,
    Ungodly = 1 << 16,
    ThereIsNoLimit = 1 << 17,
    OrIsThere = 1 << 18,
    ThisIsBlasphemy = 1 << 19,
    JustOneMoreSetting = 1 << 20,
    YourComputerIsASmokeMachine = 1 << 21,
}

impl Default for AudioBufferSize {
    fn default() -> Self {
        Self::Huge
    }
}

const SAMPLE_RATE_AUDIO: u32 = 44_100;
const SAMPLE_RATE_VIDEO: u32 = 48_000;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, EnumIter, Display)]
#[repr(u32)]
pub enum SampleRate {
    Audio = SAMPLE_RATE_AUDIO,
    Video = SAMPLE_RATE_VIDEO,
    Audio2 = SAMPLE_RATE_AUDIO * 2,
    Video2 = SAMPLE_RATE_VIDEO * 2,
    Audio4 = SAMPLE_RATE_AUDIO * 4,
    Video4 = SAMPLE_RATE_VIDEO * 4,
    Audio8 = SAMPLE_RATE_AUDIO * 8,
    Video8 = SAMPLE_RATE_VIDEO * 8,
}

pub fn analyze_audio(
    mut audio_consumer: rtrb::Consumer<f32>,
    ui_sender: crossbeam_channel::Sender<AnalysisData>,
    audio_buffer_size: Arc<AtomicUsize>,
) {
    let buffer_increment = 2048;
    let mut buffer_size = audio_buffer_size.load(Ordering::Relaxed);
    let mut fft_size = buffer_size / 2; // assuming 2ch audio
    let mut audio_buffer: VecDeque<f32> = vec![0.0f32; buffer_size].into();

    let mut fft_planner = RealFftPlanner::new();
    let mut fft = fft_planner.plan_fft_forward(fft_size);
    let mut spectrum = fft.make_output_vec();

    loop {
        let start = Instant::now();

        let new_buffer_size = audio_buffer_size.load(Ordering::Relaxed);
        if new_buffer_size != buffer_size {
            buffer_size = new_buffer_size;
            fft_size = buffer_size / 2;
            audio_buffer.resize(buffer_size, 0.0);
            fft = fft_planner.plan_fft_forward(fft_size);
            spectrum = fft.make_output_vec();
        }

        let mut samples_to_read = audio_consumer.slots();
        while samples_to_read > buffer_increment {
            match audio_consumer.read_chunk(buffer_increment) {
                Ok(chunk) => {
                    //for (dest, src) in audio_buffer.iter_mut().zip(chunk.into_iter()) {
                    //    *dest = src;
                    //}

                    audio_buffer.extend(chunk.into_iter());
                    samples_to_read -= buffer_increment;
                }
                Err(_) => {}
            }
        }

        if audio_buffer.len() > buffer_size {
            audio_buffer.drain(..(audio_buffer.len() - buffer_size));
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

        let end = Instant::now();

        match ui_sender.send(AnalysisData {
            rms_meter: (left_rms, right_rms),
            spectrum: spectrum.iter().skip(1).map(|c| c.re.abs()).collect(),
            elapsed: end - start,
        }) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("analysis thread: {}", e);
                return;
            }
        }
    }
}

pub fn build_audio_input_stream(
    device_index: usize,
    sample_rate: SampleRate,
    mut producer: rtrb::Producer<f32>,
) -> (cpal::Host, cpal::Stream, cpal::StreamConfig) {
    let host = cpal::default_host();

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

    let config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(sample_rate as u32),
        buffer_size: cpal::BufferSize::Default,
    };

    eprintln!("  {:?}", config);

    //eprintln!("using {}@{}ch", format, channels);

    let data_callback = move |data: &[f32], _info: &cpal::InputCallbackInfo| {
        assert!(RINGBUFFER_CAPACITY >= data.len());
        match producer.write_chunk_uninit(data.len()) {
            Ok(chunk) => {
                chunk.fill_from_iter(data.iter().copied());
            }
            Err(e) => {
                eprintln!("audio thread: {}", e);
            }
        }
    };

    let err_fn = |err| eprintln!("input stream error: {}", err);

    let stream = device
        .build_input_stream(&config, data_callback, err_fn, None)
        .expect("failed to build input stream");

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

pub fn frequencies(num_bins: u32, sample_rate: u32) -> Vec<f32> {
    let interval = sample_rate as f32 / (num_bins * 2) as f32;
    (1..=num_bins).map(|k| k as f32 * interval).collect()
}

pub fn frequency_resolution(fft_buffer_size: u32, sample_rate: u32) -> f32 {
    fft_buffer_size as f32 / sample_rate as f32
}

pub fn tilt(slope: f32, spectrum: &mut [f32]) {
    let alpha = slope / (20.0 * 2.0f32.log10());
    spectrum.iter_mut().enumerate().for_each(|(i, s)| {
        let gain = ((i + 1) as f32).powf(alpha);
        *s *= gain;
    })
}

pub fn smooth_lin(window_size: usize, spectrum: &mut [f32]) {
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

fn make_log_bands(start_freq: f32, max_freq: f32, ratio: f32) -> Vec<f32> {
    let mut bands = Vec::new();
    let mut freq = start_freq;

    while freq < max_freq {
        bands.push(freq);
        freq *= ratio;
    }

    bands.push(max_freq);

    bands
}

pub fn smooth_log(
    octave_division: f32,
    sample_rate: u32,
    spectrum: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let fft_buffer_size = (spectrum.len() - 1) as u32 * 2;
    let frequency_ratio = 2.0f32.powf(1.0 / octave_division);
    let frequency_res = frequency_resolution(fft_buffer_size, sample_rate);
    let band_edges = make_log_bands(20.0, sample_rate as f32 / 2.0, frequency_ratio);

    let mut smooth_spectrum = Vec::with_capacity(band_edges.len() - 1);
    let mut frequencies = Vec::with_capacity(band_edges.len() - 1);

    for bins in band_edges.windows(2) {
        let bin = bins[0];
        let next_bin = bins[1];
        let start = ((bin / frequency_res).floor() as usize).min(spectrum.len() - 1);
        let end = ((next_bin / frequency_res).floor() as usize).min(spectrum.len());
        let end = if start == end { end + 1 } else { end };

        //let e = spectrum[start..end].iter().sum::<f32>() / (end - start).max(1) as f32;
        let e = spectrum[start..end]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f32;

        smooth_spectrum.push(e);
        frequencies.push(bin);
    }

    (smooth_spectrum, frequencies)
}
//
//pub fn translate
