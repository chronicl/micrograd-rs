## micrograd in rust

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
