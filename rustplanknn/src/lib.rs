#![feature(test)]
extern crate test;
extern crate rand;

pub mod layer;
pub mod network;

pub use layer::Layer;
pub use network::Network;
