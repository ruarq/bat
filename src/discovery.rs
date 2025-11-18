use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct DiscoveryConfig {
    port: u16,
    buffer_size: usize,
    timeout: Duration,
}

impl DiscoveryConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    pub fn with_timeout(mut self, dur: Duration) -> Self {
        self.timeout = dur;
        self
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(500),
            buffer_size: 128,
            port: 25121,
        }
    }
}

pub struct Discovery<F>
where
    F: FnMut(&UdpSocket, &[u8], SocketAddr) + Send + 'static,
{
    config: DiscoveryConfig,
    join_handle: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
    callback: Option<F>,
}

impl<F> Discovery<F>
where
    F: FnMut(&UdpSocket, &[u8], SocketAddr) + Send + 'static,
{
    pub fn new(config: DiscoveryConfig, callback: F) -> Self {
        Self {
            config,
            join_handle: None,
            running: Arc::new(AtomicBool::new(false)),
            callback: Some(callback),
        }
    }
}

impl<F> Discovery<F>
where
    F: FnMut(&UdpSocket, &[u8], SocketAddr) + Send + 'static,
{
    pub fn start(&mut self) -> Result<(), Box<dyn Error>> {
        let socket = UdpSocket::bind(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            self.config.port,
        ))?;
        socket.set_read_timeout(Some(self.config.timeout))?;

        let mut buf = vec![0u8; self.config.buffer_size];

        self.running.store(true, Ordering::Relaxed);
        let running = self.running.clone();

        let mut callback = self.callback.take().unwrap();

        let handle = thread::spawn(move || {
            //log::trace!("starting");
            while running.load(Ordering::Relaxed) {
                if let Ok((bytes_read, sender)) = socket.recv_from(&mut buf) {
                    let message = &buf[..bytes_read];
                    //log::debug!(
                    //    "incoming message from {}: '{}'",
                    //    sender,
                    //    std::str::from_utf8(message).unwrap_or("failed to parse message")
                    //);
                    callback(&socket, message, sender);
                }
            }
        });

        self.join_handle = Some(handle);

        Ok(())
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.join_handle.take() {
            handle.join().expect("failed to join discovery thread");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_start() {
        let mut discovery = Discovery::new(DiscoveryConfig::default(), |_, _, _| {});
        assert!(discovery.start().is_ok());
        discovery.stop();
    }

    #[test]
    fn discovery_connect() {
        let mut discovery = Discovery::new(DiscoveryConfig::new().with_port(2000), |_, m, _| {
            assert_eq!(b"hello", m);
        });

        discovery.start().unwrap();

        let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
        socket.set_broadcast(true).unwrap();
        let bytes_sent = socket.send_to(b"hello", "127.0.0.1:2000").unwrap_or(0);

        assert_eq!(5, bytes_sent);

        discovery.stop();
    }
}
