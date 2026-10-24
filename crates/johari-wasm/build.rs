// HACK: This can go once <https://github.com/SFBdragon/talc/issues/56> is fixed.
fn main() {
    println!("cargo::rustc-link-arg-cdylib=--max-memory=2147483648");
}
