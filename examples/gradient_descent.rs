use micrograd::{Bump, Value, MLP};

fn main() {
    let bump = Bump::new();
    let mlp = MLP::new(3, &[4, 4, 1], &bump);

    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0];

    let now = std::time::Instant::now();

    const BATCHES: usize = 1000;
    let mut i = 0;
    let (loss, predictions) = loop {
        // Our training data should be allocated in a separate Bump here inside the loop,
        // because otherwise it's memory will never be freed. But since we are using limited
        // memory in this example, we are just using our MLP Bump here.

        let ys_pred: Vec<_> = xs
            .into_iter()
            .flat_map(|x| mlp.forward(x.iter().map(|&x| Value::new(x, &bump))))
            .collect();
        // let ys_pred_f = ys_pred.iter().map(|y| y.data()).collect::<Vec<_>>();

        let ys = ys.iter().map(|&y| Value::new(y, &bump));
        let loss = ys_pred
            .iter()
            .copied()
            .zip(ys)
            .map(|(y_pred, y)| (y_pred - y).powi(2))
            .sum::<Value>();

        mlp.parameters().for_each(|p| p.clear_grad());
        loss.backward();
        mlp.parameters().for_each(|p| p.adjust(-0.2));

        i += 1;
        if i == BATCHES {
            break (
                loss.data(),
                ys_pred.iter().map(|y| y.data()).collect::<Vec<_>>(),
            );
        }
    };
    println!("Elapsed: {:?}", now.elapsed());
    println!("Loss: {}", loss);
    println!("Predictions: {:?}", predictions);
}
