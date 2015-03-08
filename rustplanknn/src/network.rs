use layer::Layer;
 
pub struct Network {
    layers: Vec<Layer<f32>>
}

impl Network {
    pub fn new() -> Network {
        //let ls = vec![Layer::new(62500, 400), Layer::new(400, 400), Layer::new(400, 2)];
        let ls = vec![Layer::new(3, 400), Layer::new(400, 400), Layer::new(400, 2)];
        Network {layers: ls}
    }

    pub fn classify(&mut self, data: &Vec<f32>) -> Vec<f32> {
        let mut res = data.clone();
        for layer in self.layers.iter_mut() {
            res = layer.forward(&res);
        }
        res.clone()
    }

    pub fn train(&mut self, data: &Vec<f32>, exp_res: &Vec<f32>) -> Vec<f32> {
        let res = self.classify(data);
        let mut err_in: Vec<f32> = res.iter().zip(exp_res.iter()).map(|(&i, &j)| i - j).collect();
        for layer in self.layers.iter_mut().rev() {
            let (e, err_grad) = layer.error(&err_in);
            err_in = e;

            // `learning rate` = 0.1
            let e_g: Vec<f32> = err_grad.iter().map(|&i| 0.1f32 * i).collect();
            //layer.train(&err_grad);
            layer.train(&e_g);
        }
        res
    }
}
