use std::io::stdin;

use cpal::{
    BuildStreamError, Device, Sample, SampleFormat, Stream,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

fn main() {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no input device available");

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

    std::thread::sleep(std::time::Duration::from_secs(1));

    let err_fn = |err| eprintln!("input stream error: {}", err);

    let stream = match format {
        SampleFormat::I16 => device.build_input_stream(&config, read_audio::<i16>, err_fn, None),
        SampleFormat::F32 => device.build_input_stream(&config, read_audio::<f32>, err_fn, None),
        format => panic!("unsupported format: {}", format),
    }
    .expect("couldn't build input stream");

    stream.play().expect("unable to start stream");

    let mut exit = String::new();
    stdin().read_line(&mut exit).unwrap();
}

fn read_audio<T: Sample>(data: &[T], _: &cpal::InputCallbackInfo) {
    let rms = data
        .chunks_exact(2)
        .map(|frame| frame.iter().map(|s| s.to_float_sample().to_sample::<f32>()))
        .map(|frame| frame.map(|s| s * s))
        .sum::<f32>()
        / (data.len() as f32);
    eprintln!("{}", rms);
}
