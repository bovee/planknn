extern crate planknn;

use planknn::Network;

#[test]
fn train() {
    let mut n = Network::new();
    let data = vec![0f32, 0.5f32, 1f32];
    let exp_res = vec![-0.5f32, 0.5f32];
    for _ in 0..3 {
        n.train(&data, &exp_res);
    }
}
