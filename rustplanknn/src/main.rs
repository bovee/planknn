#![feature(test)]
extern crate test;
extern crate planknn;

use test::Bencher;
use planknn::Network;

fn main() {
    let mut n = Network::new();
    let data = vec![0f32, 0.5f32, 1f32];
    let exp_res = vec![-0.5f32, 0.5f32];
    println!("{:?}", n.classify(&data));
    for _ in 0..3 {
        n.train(&data, &exp_res);
    }
    println!("{:?}", n.classify(&data));
}

//#[bench]
//fn bench(b: &mut Bencher) {
//    b.iter(|| {
//        let mut n = Network::new();
//        let data = vec![0f32, 0.5f32, 1f32];
//        let exp_res = vec![-0.5f32, 0.5f32];
//        n.train(&data, &exp_res);
//    });
//}
