use bytemuck::NoUninit;
use std::ops::{Add, AddAssign};

/// Represents a RGB color value.
#[derive(Debug, Clone, Copy, PartialEq, NoUninit)]
#[repr(C, packed)]
pub struct Color(u8, u8, u8);

impl Color {
    pub const WHITE: Self = Self::new(255, 255, 255);
    pub const BLACK: Self = Self::new(0, 0, 0);

    pub const RED: Self = Self::new(255, 0, 0);
    pub const GREEN: Self = Self::new(0, 255, 0);
    pub const BLUE: Self = Self::new(0, 0, 255);

    pub const YELLOW: Self = Self::new(255, 255, 0);
    pub const MAGENTA: Self = Self::new(255, 0, 255);
    pub const CYAN: Self = Self::new(0, 255, 255);

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self(r, g, b)
    }

    pub const fn from_hex(rgb: u32) -> Self {
        Self(
            ((rgb >> 16) & 0xFF) as u8,
            ((rgb >> 8) & 0xFF) as u8,
            (rgb & 0xFF) as u8,
        )
    }

    pub const fn r(&self) -> u8 {
        self.0
    }

    pub const fn g(&self) -> u8 {
        self.1
    }

    pub const fn b(&self) -> u8 {
        self.2
    }

    pub const fn rgb(&self) -> (u8, u8, u8) {
        (self.r(), self.g(), self.b())
    }

    fn interpolate_component(x: f32, lhs: u8, rhs: u8) -> u8 {
        if lhs < rhs {
            lhs + ((rhs - lhs) as f32 * x) as u8
        } else {
            lhs - ((lhs - rhs) as f32 * x).ceil() as u8
        }
    }

    pub fn interpolate(x: f32, lhs: Self, rhs: Self) -> Self {
        assert!((0.0f32..=1.0f32).contains(&x));

        Self(
            Self::interpolate_component(x, lhs.0, rhs.0),
            Self::interpolate_component(x, lhs.1, rhs.1),
            Self::interpolate_component(x, lhs.2, rhs.2),
        )
    }

    pub fn scaled(self, s: f32) -> Self {
        Self(
            (self.0 as f32 * s) as u8,
            (self.1 as f32 * s) as u8,
            (self.2 as f32 * s) as u8,
        )
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let r = self.0.saturating_add(rhs.0);
        let g = self.1.saturating_add(rhs.1);
        let b = self.2.saturating_add(rhs.2);
        Self(r, g, b)
    }
}

impl AddAssign for Color {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[derive(Debug, Clone)]
pub struct Gradient {
    colors: Vec<Color>,
}

impl Gradient {
    pub fn new(colors: Vec<Color>) -> Self {
        Self { colors }
    }

    pub fn at(&self, mut t: f32) -> Color {
        t = t.clamp(0.0, 1.0);

        let len = self.colors.len();
        let max_index = if len == 0 { 1 } else { len - 1 };
        let points = self
            .colors
            .iter()
            .enumerate()
            .map(|(i, color)| ((i as f32 / max_index as f32), color));

        // find the closest two points p1, p2 with p1 <= t <= p2
        let (p1, prev) = points
            .clone()
            .rfind(|(x, _)| *x <= t)
            .expect("failed to find previous point");

        let (p2, next) = points
            .clone()
            .find(|(x, _)| t <= *x)
            .expect("failed to find next point");

        if p1 == p2 {
            return *prev;
        }

        let point = (t - p1) / (p2 - p1);

        Color::interpolate(point, *prev, *next)
    }

    pub fn scaled(mut self, s: f32) -> Self {
        for color in &mut self.colors {
            *color = color.scaled(s);
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_hex() {
        assert_eq!((255, 0, 0), Color::from_hex(0xFF0000).rgb());
        assert_eq!((0, 255, 0), Color::from_hex(0x00FF00).rgb());
        assert_eq!((0, 0, 255), Color::from_hex(0x0000FF).rgb());
    }

    #[test]
    fn add_assign() {
        assert_eq!(
            (255, 255, 0),
            (Color::new(255, 0, 0) + Color::new(0, 255, 0)).rgb()
        );

        assert_eq!(
            (0, 100, 255),
            (Color::new(0, 25, 250) + Color::new(0, 75, 255)).rgb()
        );
    }

    #[test]
    fn interpolation() {
        assert_eq!(255, Color::interpolate_component(0.0, 255, 0));
        assert_eq!(0, Color::interpolate_component(1.0, 255, 0));
        assert_eq!(127, Color::interpolate_component(0.5, 255, 0));

        assert_eq!(0, Color::interpolate_component(0.0, 0, 255));
        assert_eq!(255, Color::interpolate_component(1.0, 0, 255));
        assert_eq!(127, Color::interpolate_component(0.5, 0, 255));

        assert_eq!(
            Color::RED,
            Color::interpolate(0.0, Color::RED, Color::GREEN)
        );
        assert_eq!(
            Color::new(127, 127, 0),
            Color::interpolate(0.5, Color::RED, Color::GREEN)
        );
        assert_eq!(
            Color::GREEN,
            Color::interpolate(1.0, Color::RED, Color::GREEN)
        );
    }

    #[test]
    fn scaled_overflow() {
        assert_eq!(Color::RED, Color::RED.scaled(2.0));
    }

    #[test]
    fn linear_gradient_2colors() {
        let gradient = Gradient::new(vec![Color::RED, Color::GREEN]);
        assert_eq!(Color::RED, gradient.at(0.0));
        assert_eq!(Color::new(127, 127, 0), gradient.at(0.5));
        assert_eq!(Color::GREEN, gradient.at(1.0));
    }

    #[test]
    fn linear_gradient_3colors() {
        let gradient = Gradient::new(vec![Color::RED, Color::GREEN, Color::BLUE]);
        assert_eq!(Color::RED, gradient.at(0.0));
        assert_eq!(Color::GREEN, gradient.at(0.5));
        assert_eq!(Color::BLUE, gradient.at(1.0));

        // this is from a bug, where every color t > 0.5 would equal the last color in the gradient
        assert_ne!(Color::BLUE, gradient.at(0.6));
    }
}
