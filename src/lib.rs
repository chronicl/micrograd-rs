/*!
My take on a port of micrograd to Rust. Using a bump allocator to make `Value`s Copy
and thus allow users to use `Value`s as they would use f32s. Of course this brings along
the inconvenience of having to provide a `Bump` every time we create a new `Value`, however
operations like addition and multiplication don't require the `Bump` again.

```rust
let bump = micrograd::Bump::new();
let v0 = Value::new(1, &bump);
let v1 = Value::new(2, &bump);
let v2 = v0 + v1;
// We can still use v0 and v1 because they are Copy
let v3 = v0 * v1;
```

We are also heavily saving on allocations by using the `Bump` for everything that needs
to be heap allocated. This alone should make this implementation faster than any implementations that are based around `Rc` and `RefCell`.
A down side of this approach is that `Value`s are only deallocated when their `Bump` is deallocated.
*/

pub use bumpalo::Bump;
use std::cell::Cell;

#[derive(Clone, Copy)]
pub struct Value<'a> {
    inner: &'a ValueInner<'a>,
    pub allocator: &'a Bump,
}
pub struct ValueInner<'a> {
    data: Cell<f32>,
    grad: Cell<f32>,
    creators: &'a [Value<'a>],
    // `backward_fn` is attached by the operation that created this value and
    // will add to the gradients of it's creators, not it's own gradient.
    // Thus, when this Value is used to create multiple other Values, the gradient
    // will be accumulated when calling `backward_fn` on each of them.
    // You could instead store a list of all the `backward_fn` on this Value
    // itself, but that would require another Vec allocation for each Value.
    backward_fn: Option<&'a (dyn Fn(Value) + 'a)>,
    visited: Cell<bool>,

    label: Cell<Option<&'static str>>,
}

impl<'a> std::ops::Deref for Value<'a> {
    type Target = ValueInner<'a>;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'a> Value<'a> {
    pub fn new(value: f32, allocator: &'a Bump) -> Self {
        let inner = allocator.alloc(ValueInner {
            data: Cell::new(value),
            grad: Cell::new(0.),
            creators: allocator.alloc_slice_copy(&[]),
            backward_fn: None,
            visited: Cell::new(false),
            label: Cell::new(None),
        });
        Self { inner, allocator }
    }

    fn new_with_creators(
        value: f32,
        creators: &[Value<'a>],
        backward_fn: impl Fn(Value) + 'a,
        allocator: &'a Bump,
    ) -> Self {
        let creators = allocator.alloc_slice_copy(creators);
        let backward_fn = allocator.alloc(backward_fn);
        let inner = allocator.alloc(ValueInner {
            data: Cell::new(value),
            grad: Cell::new(0.),
            creators,
            backward_fn: Some(backward_fn),
            visited: Cell::new(false),
            label: Cell::new(None),
        });
        Self { inner, allocator }
    }

    // pub fn with_label(self, label: impl AsRef<str>) -> Self {
    //     self.label
    //         .set(Some(self.allocator.alloc_str(label.as_ref())));
    //     self
    // }

    pub fn data(self) -> f32 {
        self.data.get()
    }

    pub fn set_data(self, value: f32) {
        self.data.set(value);
    }

    pub fn grad(self) -> f32 {
        self.grad.get()
    }

    pub fn set_grad(self, grad: f32) {
        self.grad.set(grad);
    }

    pub fn add_to_grad(self, grad: f32) {
        self.set_grad(self.grad() + grad);
    }

    pub fn clear_grad(self) {
        self.visited.set(false);
        self.set_grad(0.);
    }

    pub fn adjust(self, adjustment: f32) {
        self.set_data(self.data() + adjustment * self.grad());
    }

    pub fn backward(self) {
        self.grad.set(1.);
        let mut stack = vec![self];
        while let Some(value) = stack.pop() {
            if value.visited.get() {
                continue;
            }
            value.visited.set(true);

            if let Some(backward_fn) = value.backward_fn {
                backward_fn(value);
                stack.extend_from_slice(value.creators);
            }
        }
    }

    pub fn tanh(self) -> Self {
        let value = self.data().tanh();
        let creators = &[self];
        let backward_fn = move |out: Value| {
            self.add_to_grad((1. - value * value) * out.grad());
        };
        Value::new_with_creators(value, creators, backward_fn, self.allocator)
    }

    pub fn powi(self, power: i32) -> Self {
        let value = self.data().powi(power);
        let creators = &[self];
        let backward_fn = move |out: Value| {
            self.add_to_grad(power as f32 * self.data().powi(power - 1) * out.grad());
        };
        Value::new_with_creators(value, creators, backward_fn, self.allocator)
    }

    pub fn pow(self, power: f32) -> Self {
        let value = self.data().powf(power);
        let creators = &[self];
        let backward_fn = move |out: Value| {
            self.add_to_grad(power * self.data().powf(power - 1.) * out.grad());
        };
        Value::new_with_creators(value, creators, backward_fn, self.allocator)
    }
}

/// Value<'a> + Value<'b> = Value<'a>
/// same for all the other binary operations
impl<'a, 'b> std::ops::Add<Value<'b>> for Value<'a>
where
    'b: 'a,
{
    type Output = Self;

    fn add(self, other: Value<'b>) -> Self {
        let value = self.data() + other.data();
        let creators = &[self, other];
        let backward_fn = move |out: Value| {
            self.add_to_grad(out.grad());
            other.add_to_grad(out.grad());
        };
        Value::new_with_creators(value, creators, backward_fn, self.allocator)
    }
}

impl<'a, 'b> std::ops::Sub<Value<'b>> for Value<'a>
where
    'b: 'a,
{
    type Output = Self;

    fn sub(self, other: Value<'b>) -> Self {
        let value = self.data() - other.data();
        let creators = &[self, other];
        let backward_fn = move |out: Value| {
            self.add_to_grad(out.grad());
            other.add_to_grad(-out.grad());
        };
        Value::new_with_creators(value, creators, backward_fn, self.allocator)
    }
}

impl<'a, 'b> std::ops::Mul<Value<'b>> for Value<'a>
where
    'b: 'a,
{
    type Output = Self;

    fn mul(self, other: Value<'b>) -> Self {
        let value = self.data() * other.data();
        let creators = &[self, other];
        let backward_fn = move |out: Value| {
            self.add_to_grad(other.data() * out.grad());
            other.add_to_grad(self.data() * out.grad());
        };
        Value::new_with_creators(value, creators, backward_fn, self.allocator)
    }
}

impl std::ops::Neg for Value<'_> {
    type Output = Self;

    fn neg(self) -> Self {
        let value = -self.data();
        let creators = &[self];
        let backward_fn = move |out: Value| {
            self.add_to_grad(-out.grad());
        };
        Value::new_with_creators(value, creators, backward_fn, self.allocator)
    }
}

impl std::fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("label", &self.label.get())
            .field("data", &self.data())
            .field("grad", &self.grad())
            .finish()
    }
}

pub struct Neuron<'a> {
    weight: Vec<Value<'a>>,
    bias: Value<'a>,
}

pub struct Layer<'a> {
    neurons: Vec<Neuron<'a>>,
}

pub struct MLP<'a> {
    layers: Vec<Layer<'a>>,
}

impl<'a> Neuron<'a> {
    pub fn new(input_size: usize, allocator: &'a Bump) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let weight = (0..input_size)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0), allocator))
            .collect();
        let bias = Value::new(rng.gen_range(-1.0..1.0), allocator);
        Self { weight, bias }
    }

    pub fn forward<'b>(&self, input: impl IntoIterator<Item = Value<'b>>) -> Value<'b>
    where
        'a: 'b,
    {
        let mut sum = self.bias;
        for (w, x) in self.weight.iter().zip(input) {
            // Here it is essential that the operations happen in this order.
            // x has lifetime 'b and w and the initial sum have lifetime 'a.
            // We want our output to have lifetime 'b and results of operations
            // take on the lifetime of the left operand, so we need to make
            // sure x is on the left.
            sum = x * *w + sum;
        }
        sum.tanh()
    }

    pub fn parameters(&'a self) -> impl Iterator<Item = Value<'a>> {
        self.weight
            .iter()
            .copied()
            .chain(std::iter::once(self.bias))
    }
}

impl<'a> Layer<'a> {
    pub fn new(input_size: usize, output_size: usize, allocator: &'a Bump) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::new(input_size, allocator))
            .collect();
        Self { neurons }
    }

    pub fn forward<'b>(
        &'a self,
        input: impl IntoIterator<Item = Value<'b>> + Clone,
    ) -> impl Iterator<Item = Value<'b>>
    where
        'a: 'b,
    {
        self.neurons.iter().map(move |n| n.forward(input.clone()))
    }

    pub fn parameters(&'a self) -> impl Iterator<Item = Value<'a>> {
        self.neurons.iter().flat_map(|n| n.parameters())
    }
}

impl<'a> MLP<'a> {
    pub fn new(input_size: usize, hidden_sizes: &[usize], allocator: &'a Bump) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;
        for &size in hidden_sizes {
            layers.push(Layer::new(prev_size, size, allocator));
            prev_size = size;
        }
        Self { layers }
    }

    pub fn forward<'b>(
        &'a self,
        input: impl IntoIterator<Item = Value<'b>> + Clone,
    ) -> Vec<Value<'b>>
    where
        'a: 'b,
    {
        let mut output = Vec::new();
        let mut input = self.layers[0].forward(input).collect::<Vec<_>>();

        if self.layers.len() < 2 {
            return input;
        }

        for layer in self.layers[1..].iter() {
            output.extend(layer.forward(input.iter().copied()));
            input.clear();
            input.append(&mut output);
        }
        input
    }

    pub fn parameters(&'a self) -> impl Iterator<Item = Value<'a>> {
        self.layers.iter().flat_map(|l| l.parameters())
    }
}

impl<'a> std::iter::Sum for Value<'a> {
    fn sum<I: Iterator<Item = Value<'a>>>(mut iter: I) -> Self {
        // Todo: This unwrap is obviously not good, but we can't get the Bump in here
        // for creating a new Value.
        let mut sum = iter.next().unwrap();
        for x in iter {
            sum = sum + x;
        }
        sum
    }
}

#[test]
fn test_value() {
    let bump = Bump::new();
    let mut v = Vec::new();
    v.push(Value::new(1., &bump));
    v.push(Value::new(2., &bump));
    v.push(v[0] + v[1]);
    v[2].backward();
    assert_eq!(v[0].grad(), 1.);
    assert_eq!(v[1].grad(), 1.);
    v.iter().for_each(|v| v.clear_grad());

    v.push(v[0] - v[1]);
    v[3].backward();
    assert_eq!(v[0].grad(), 1.);
    assert_eq!(v[1].grad(), -1.);
    v.iter().for_each(|v| v.clear_grad());

    v.push(v[0] * v[1]);
    v[4].backward();
    assert_eq!(v[0].grad(), 2.);
    assert_eq!(v[1].grad(), 1.);
    v.iter().for_each(|v| v.clear_grad());

    v.push(v[0].tanh());
    v[5].backward();
    assert_eq!(v[0].grad(), 1. - v[5].data() * v[5].data());
    v.iter().for_each(|v| v.clear_grad());

    v.push(v[2] + v[3]);
    v[6].backward();
    assert_eq!(v[0].grad(), 2.);
    assert_eq!(v[1].grad(), 0.);
    assert_eq!(v[2].grad(), 1.);
    assert_eq!(v[3].grad(), 1.);
    v.iter().for_each(|v| v.clear_grad());
}
