use std::io::{self, Write};
use std::time::Instant;

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::inference::generate;
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::train::train_step;

fn names() -> Vec<&'static str> {
    vec![
        "emma",
        "olivia",
        "ava",
        "isabella",
        "sophia",
        "mia",
        "charlotte",
        "amelia",
        "harper",
        "evelyn",
        "liam",
        "noah",
        "oliver",
        "elijah",
        "william",
        "james",
        "benjamin",
        "lucas",
        "mason",
        "ethan",
        "aiden",
        "logan",
        "jackson",
        "sebastian",
        "mateo",
        "jack",
        "owen",
        "theodore",
        "samuel",
        "henry",
        "leo",
        "luke",
        "jayden",
        "gabriel",
        "landon",
        "anthony",
        "dylan",
        "carter",
        "julian",
        "layla",
        "zoe",
        "penelope",
        "lily",
        "eleanor",
        "nora",
        "luna",
        "hazel",
        "aurora",
        "chloe",
        "aria",
        "grace",
        "zoey",
        "riley",
        "violet",
        "nova",
        "camille",
        "claire",
        "isla",
        "sofia",
        "scarlett",
        "elena",
        "alice",
        "savannah",
        "daisy",
        "audrey",
        "ruby",
        "stella",
        "naomi",
        "adeline",
        "ryan",
        "caleb",
        "eli",
        "christian",
        "josiah",
        "nathan",
        "wyatt",
        "andrew",
        "joshua",
        "christopher",
        "lincoln",
        "thomas",
        "ezra",
        "hudson",
        "daniel",
        "nicholas",
        "peter",
        "john",
        "levi",
        "ian",
        "axel",
        "cole",
        "beau",
        "felix",
        "maya",
        "nadia",
        "iris",
        "june",
        "vera",
    ]
}

fn main() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let mut rng = Rng::new(42);
    let docs = names();
    let vocab = build_vocab(&docs);
    let mut model = Model::new(vocab.size(), &mut rng, mc, &tc);

    println!(
        "microgpt-rs  vocab={}  params={}  layers={}  embd={}  heads={}",
        vocab.size(),
        model.param_count(),
        mc.n_layer,
        mc.n_embd,
        mc.n_head
    );
    println!("Training {} steps\n", tc.n_steps);

    let t0 = Instant::now();

    for step in 0..tc.n_steps {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        let loss = train_step(&mut model, &tokens, step, &tc);

        if step % 100 == 0 || step == tc.n_steps - 1 {
            let samples = generate(
                &model.sd,
                &vocab,
                &mut rng,
                &mc,
                tc.n_samples,
                tc.temperature,
                "",
            );
            print!(
                "step {:>5}  loss={:.4}  t={:.2}s  | ",
                step,
                loss,
                t0.elapsed().as_secs_f32()
            );
            for (i, s) in samples.iter().enumerate() {
                print!("{}{}", s, if i < samples.len() - 1 { "  " } else { "" });
            }
            println!();
            io::stdout().flush().unwrap();
        }
    }

    println!("\nDone in {:.3}s", t0.elapsed().as_secs_f32());
}
