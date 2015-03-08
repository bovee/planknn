use std::num::Float;
use rand;
use rand::Rng;
use test::Bencher;

#[test]
fn test_create() {
    let l = Layer::<f32>::new(3, 2);
    assert_eq!(l.weights.len(), 6);
}

#[test]
fn test_forward() {
    let mut l = Layer::<f32>::new(3, 2);
    l.weights = vec![0f32, 0f32, 0f32, 1f32, 1f32, 1f32];
    let r = l.forward(&vec![10f32, 20f32, 30f32]);
    assert_eq!(r.len(), 2);
    assert_eq!(l.f_input, vec![0f32, 60f32]);
    assert_eq!(r, vec![0f32, 1f32]);
}

#[bench]
fn bench_forward(b: &mut Bencher) {
    let mut l = Layer::<f32>::new(100, 100);
    b.iter(|| {
        l.forward(&vec![1f32; 100]);
    });
}

#[bench]
fn bench_backprop(b: &mut Bencher) {
    let mut l = Layer::<f32>::new(100, 100);
    l.forward(&vec![1f32; 100]);
    b.iter(|| {
        l.error(&vec![0f32; 100]);
    });
}

#[test]
fn test_error() {
    let mut l = Layer::<f32>::new(3, 2);
    l.weights = vec![0f32, 0f32, 0f32, 1f32, 1f32, 1f32];
    l.forward(&vec![10f32, 20f32, 30f32]);

    let (err_in, err_grad) = l.error(&vec![0f32, 0.5f32]);
    assert_eq!(err_in.len(), 3);
    assert_eq!(err_grad.len(), 6);
}

pub struct Layer<T> where T: Float {
    n_inputs: usize,
    n_outputs: usize,
    input: Vec<T>,
    f_input: Vec<T>,
    pub weights: Vec<T>,
    func: Box<Fn(T) -> T>,
    dfunc: Box<Fn(T) -> T>,
}

impl Layer<f32> { //where T: Float + rand::Rand
    pub fn new(n_inputs: usize, n_outputs: usize) -> Layer<f32> {
        let mut rng = rand::thread_rng();
        let w: Vec<f32> = (0..n_inputs * n_outputs).map(|_| rng.gen_range(-0.1, 0.1)).collect();

        let f = |v: f32| v.tanh();
        let df = |v: f32| 1f32 - v.tanh().powi(2i32);
        Layer {n_inputs: n_inputs, n_outputs: n_outputs, weights: w.clone(),
               input: Vec::new(), f_input: Vec::new(),
               func: Box::new(f),
               dfunc: Box::new(df),
              }
    }

    pub fn forward(&mut self, data: &Vec<f32>) -> Vec<f32> {
        self.input = data.clone();
        let mut f_input = vec![0f32; self.n_outputs];
        // calculate the dot product with the data
        for j in 0..self.n_outputs {
            for i in 0..self.n_inputs {
                f_input[j] += data[i] * self.weights[j * self.n_inputs + i];
            }
        }

        self.f_input = f_input.clone();
        // apply tanh over the results and return it
        let mut output = vec![0f32; self.n_outputs];
        for j in (0..self.n_outputs) {
            output[j] = (*self.func)(f_input[j]);
        }
        output
    }

    pub fn error(&self, error_out: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        // state is dependent on last invocation of forward
        let mut error_in_f = vec![0f32; self.n_outputs];
        for j in (0..self.n_outputs) {
            error_in_f[j] = (*self.dfunc)(self.f_input[j]) * error_out[j];
        }

        let mut error_gradient = vec![0f32; self.n_inputs * self.n_outputs];
        let mut error_in = vec![0f32; self.n_inputs];

        for i in 0..self.n_inputs {
            for j in 0..self.n_outputs {
                // error_in = dot(error_in_f, self.weights.T)
                error_in[i] += error_in_f[j] * self.weights[j * self.n_inputs + i];
                // error_gradient = outer(self.input, error_in_f)
                error_gradient[j * self.n_inputs + i] = self.input[i] * error_in_f[j];
            }
        }
        (error_in, error_gradient)
    }

    pub fn train(&mut self, weight_gradient: &Vec<f32>) {
        for i in 0..self.n_inputs * self.n_outputs {
            self.weights[i] -= weight_gradient[i];
        }
    }
}
