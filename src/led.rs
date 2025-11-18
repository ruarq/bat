use crate::gradient::{Color, Gradient};

pub struct LedStrip {
    pub data: Vec<Color>,
    pub gradient: Gradient,
    pub dimmed: f32,
}

impl LedStrip {
    pub fn with_size(num_leds: usize) -> Self {
        Self {
            data: vec![Color::new(0, 255, 0); num_leds],
            gradient: Gradient::new(vec![
                Color::new(100, 0, 100),
                Color::new(200, 0, 50),
                Color::MAGENTA,
            ]),
            dimmed: 0.2,
        }
    }

    pub fn amp(&mut self, meter: f32) {
        assert!(meter >= 0.0 && meter <= 1.0);

        let lit_leds = (meter * self.data.len() as f32) as usize;

        let data_len = self.data.len();
        for (i, rgb) in self.data.iter_mut().enumerate() {
            let color = self.gradient.at(i as f32 / data_len as f32);
            *rgb = if i < lit_leds {
                color
            } else {
                color.scaled(self.dimmed)
            };
        }
    }

    pub fn biamp(&mut self, meter: f32) {
        assert!(meter >= 0.0 && meter <= 1.0);

        // TODO: simplify this calculation
        let lit_leds = (meter * self.data.len() as f32) as usize / 2;
        let center_left = self.data.len() / 2 - 1;
        let center_right = self.data.len() / 2;
        let start = if lit_leds <= center_left {
            center_left - lit_leds
        } else {
            0
        };
        let end = center_right + lit_leds;

        let data_len = self.data.len();
        for (i, rgb) in self.data.iter_mut().enumerate() {
            let color = self.gradient.at(i as f32 / data_len as f32);
            *rgb = if i >= start && i < end {
                color
            } else {
                color.scaled(self.dimmed)
            };
        }
    }

    pub fn amps(&mut self, amps: &[f32]) {
        assert_eq!(self.data.len(), amps.len());

        let data_len = self.data.len();
        for (i, (rgb, amp)) in self.data.iter_mut().zip(amps).enumerate() {
            let color = self.gradient.at(i as f32 / data_len as f32);
            *rgb = if *amp > 0.0 {
                color
            } else {
                color.scaled(self.dimmed)
            };
        }
    }

    pub fn biamps(&mut self, amps: &[f32]) {
        assert_eq!(self.data.len(), amps.len());

        let amps_len = self.data.len() / 2;
        let amps_it = amps
            .chunks(2)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32);

        for (i, (rgb, amp)) in self.data[..amps_len]
            .iter_mut()
            .zip(amps_it.clone().rev())
            .enumerate()
        {
            let color = self.gradient.at((amps_len - i) as f32 / amps_len as f32);
            *rgb = if amp > 0.0 {
                color
            } else {
                color.scaled(self.dimmed)
            };
        }

        for (i, (rgb, amp)) in self.data[amps_len..].iter_mut().zip(amps_it).enumerate() {
            let color = self.gradient.at(i as f32 / amps_len as f32);
            *rgb = if amp > 0.0 {
                color
            } else {
                color.scaled(self.dimmed)
            };
        }
    }
}
