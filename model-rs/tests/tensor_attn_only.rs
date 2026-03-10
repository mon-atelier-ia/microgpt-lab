//! Test attention backward with realistic Q/K/V setup.

use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::tensor::{Shape, Tensor};
use microgpt_rs::tensor_model::TensorModel;
use microgpt_rs::value::Value;
use microgpt_rs::ops::{linear, rmsnorm, softmax};

#[test]
fn attention_only_grads_match() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let docs = vec!["emma", "olivia"];
    let vocab = build_vocab(&docs);
    let tokens = tokenize("emma", &vocab, mc.block_size);
    let tid = tokens[0];
    let e = mc.n_embd;
    let n_head = mc.n_head;
    let head_dim = e / n_head;

    let mut rng_s = Rng::new(42);
    let sm = Model::new(vocab.size(), &mut rng_s, mc, &tc);
    let mut rng_t = Rng::new(42);
    let tm = TensorModel::new(vocab.size(), &mut rng_t, mc, &tc);

    // === SCALAR: embed → rmsnorm → rmsnorm → QKV → attention → sum ===
    let lw = &sm.sd.layers[0];
    let sx: Vec<Value> = sm.sd.wte[tid].iter().zip(&sm.sd.wpe[0]).map(|(t, p)| t.add(p)).collect();
    let sx = rmsnorm(&sx);
    let sx = rmsnorm(&sx);
    let sq = linear(&sx, &lw.attn_wq);
    let sk = linear(&sx, &lw.attn_wk);
    let sv = linear(&sx, &lw.attn_wv);

    let scale = (head_dim as f64).sqrt();
    let mut sx_attn: Vec<Value> = Vec::with_capacity(e);
    for h in 0..n_head {
        let hs = h * head_dim;
        let sq_h = &sq[hs..hs + head_dim];
        let sk_h = &sk[hs..hs + head_dim];
        let dot = sq_h.iter().zip(sk_h)
            .map(|(qi, ki)| qi.mul(ki))
            .reduce(|a, b| a.add(&b)).unwrap()
            .mul_f64(1.0 / scale);
        let attn_w = softmax(&[dot]);
        for j in 0..head_dim {
            sx_attn.push(attn_w[0].mul(&sv[hs + j]));
        }
    }

    // Simple loss: sum of attention output
    let s_loss = sx_attn.iter().skip(1).fold(sx_attn[0].clone(), |a, b| a.add(b));
    s_loss.backward();
    let s_wpe: Vec<f64> = sm.sd.wpe[0].iter().map(|v| v.grad()).collect();
    let s_wv0: Vec<f64> = lw.attn_wv[0].iter().map(|v| v.grad()).collect();

    // === TENSOR: same path ===
    let tlw = &tm.sd.layers[0];
    let tok = tm.sd.wte.select_row(tid);
    let pos = tm.sd.wpe.select_row(0);
    let tx = tok.add(&pos).rmsnorm().rmsnorm();
    let tq = tx.matmul_rhs_t(&tlw.attn_wq);
    let tk = tx.matmul_rhs_t(&tlw.attn_wk);
    let tv = tx.matmul_rhs_t(&tlw.attn_wv);

    let mut head_outs: Vec<Tensor> = Vec::new();
    for h in 0..n_head {
        let hs = h * head_dim;
        let he = hs + head_dim;
        let q_h = tq.select_cols(hs, he);
        let k_h = tk.select_cols(hs, he);

        // attn_logits = q_h @ k_h^T / scale → [1,1]
        let attn_logits = q_h.matmul_rhs_t(&k_h).scale(1.0 / scale);
        // Note: k_h is [1, head_dim]. matmul_rhs_t: [1,hd] @ [1,hd]^T = [1,1]. Correct.
        let attn_w = attn_logits.softmax();

        let v_h = tv.select_cols(hs, he);
        // With stack_rows on single row: [1, head_dim]
        let v_mat = Tensor::stack_rows(&[v_h]);
        let head_out = attn_w.matmul(&v_mat); // [1,1] @ [1,hd] = [1,hd]
        head_outs.push(head_out);
    }
    let x_attn = Tensor::cat_cols(&head_outs);
    let ones = Tensor::new(vec![1.0; e], Shape(1, e));
    let t_loss = x_attn.matmul_rhs_t(&ones);
    t_loss.backward();
    let t_wpe = tm.sd.wpe.row_grad(0);
    let t_wv0 = tlw.attn_wv.row_grad(0);

    let max_wpe: f64 = s_wpe.iter().zip(&t_wpe).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let max_wv: f64 = s_wv0.iter().zip(&t_wv0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Attention-only max wpe diff: {max_wpe:.2e}");
    println!("Attention-only max wv diff: {max_wv:.2e}");
    println!("Scalar wpe[0..4]: {:?}", &s_wpe[..4]);
    println!("Tensor wpe[0..4]: {:?}", &t_wpe[..4]);
    println!("Scalar wv[0][0..4]: {:?}", &s_wv0[..4]);
    println!("Tensor wv[0][0..4]: {:?}", &t_wv0[..4]);

    assert!(max_wv < 1e-12, "wv grads differ: {max_wv}");
    assert!(max_wpe < 1e-12, "wpe grads differ: {max_wpe}");
}
