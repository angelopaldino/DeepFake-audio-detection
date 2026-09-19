"""
Analisi dei pesi di attenzione del ramo prosodico (notebook 06).

Risponde a una domanda precisa: l'attenzione si concentra sulle finestre
silenziose piu' di quanto la loro semplice proporzione giustifichi?

Riferimento di confronto. Sotto attenzione uniforme sui passi reali si avrebbe
  alpha_i = phi_i
dove alpha_i e' la massa di attenzione che cade sulle finestre mute del file i
e phi_i e' la frazione di finestre mute dello stesso file. Lo scarto
  delta_i = alpha_i - phi_i
e' quindi positivo se il modello guarda il silenzio piu' del dovuto, negativo
se lo evita, nullo se lo tratta come qualunque altra finestra.

Uso:
    python analisi_attenzione.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# PERCORSI DA IMPOSTARE
ATT_DIR     = Path("C:\\Users\\angel\\OneDrive\\Desktop\\DeepFakeDetection\\output\\Praat\\results_Part6.1\\prosody_branch_v2")   # cartella con attention_<V>_seed<k>.npz
PROSODY_DIR = Path("C:\\Users\\angel\\OneDrive\\Desktop\\DeepFakeDetection\\output\\Praat\\results_Part6.1\\prosody_branch_v2")    # cartella con standard_dev.npz e index.csv
OUT_DIR     = Path("C:\\Users\\angel\\OneDrive\\Desktop\\DeepFakeDetection\\output\\Praat\\results_Part6.1\\analisi_attenzione")

VARIANTI = ["A", "D", "E"]
SEMI     = [0, 1, 2, 3, 4]
CONFIG   = "standard"
# --------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. finestre mute del dev, dalle feature grezze ----------------------
z = np.load(PROSODY_DIR / f"{CONFIG}_dev.npz", allow_pickle=True)
X, off, utt_p = z["X"], z["offsets"].astype(np.int64), z["utt_id"]
seqs = [X[off[i]:off[i + 1]] for i in range(len(utt_p))]
# una finestra e' "muta" quando tutte le feature valgono 0 (regola del notebook 04)
mute = [np.abs(s).sum(1) == 0 for s in seqs]
lens_p = np.array([len(s) for s in seqs])

index = pd.read_csv(PROSODY_DIR / "index.csv")
index = index[index["split"] == "dev"].set_index("utt_id")
y = index.loc[utt_p, "y_bonafide"].to_numpy().astype(int)
print(f"dev: {len(utt_p)} file | finestre mediana {np.median(lens_p):.0f} | "
      f"frazione media di finestre mute {np.mean([m.mean() for m in mute]):.3f}")


def carica_attenzione(path):
    """Legge un file attention_*.npz e ne verifica la struttura."""
    z = np.load(path, allow_pickle=True)
    chiavi = list(z.keys())
    assert "attention" in chiavi, f"{path.name}: chiavi trovate {chiavi}, manca 'attention'"
    att = z["attention"]
    utt = z["utt_id"] if "utt_id" in chiavi else None
    assert len(att) == len(utt_p), f"{path.name}: {len(att)} sequenze, attese {len(utt_p)}"
    if utt is not None:
        assert (utt == utt_p).all(), f"{path.name}: ordine degli utt_id diverso da standard_dev.npz"
    # ogni sequenza deve avere la lunghezza del file corrispondente e sommare a 1
    l_att = np.array([len(a) for a in att])
    assert (l_att == lens_p).all(), \
        (f"{path.name}: lunghezze diverse dalle sequenze prosodiche "
         f"(primo disallineamento all'indice {int(np.argmax(l_att != lens_p))})")
    somme = np.array([float(np.sum(a)) for a in att])
    assert np.allclose(somme, 1.0, atol=1e-3), \
        f"{path.name}: i pesi non sommano a 1 (min {somme.min():.4f}, max {somme.max():.4f})"
    return att


def metriche(att):
    """Per ogni file: massa sul silenzio, frazione di silenzio, entropia normalizzata."""
    alpha, phi, ent = [], [], []
    for a, m, T in zip(att, mute, lens_p):
        a = np.asarray(a, dtype=np.float64)
        alpha.append(a[m].sum())
        phi.append(m.mean())
        p = np.clip(a, 1e-12, None)
        ent.append(float(-(p * np.log(p)).sum() / np.log(T)) if T > 1 else np.nan)
    return np.array(alpha), np.array(phi), np.array(ent)


def ic95(v, n_boot=2000, seed=0):
    g = np.random.default_rng(seed)
    b = [g.choice(v, len(v)).mean() for _ in range(n_boot)]
    return np.percentile(b, [2.5, 97.5])


# ---- 2. metriche per variante e seme --------------------------------------
righe, per_variante = [], {}
for V in VARIANTI:
    acc_alpha, acc_phi, acc_ent = [], [], []
    for s in SEMI:
        p = ATT_DIR / f"attention_{V}_seed{s}.npz"
        if not p.exists():
            print(f"manca {p.name}, saltato")
            continue
        alpha, phi, ent = metriche(carica_attenzione(p))
        acc_alpha.append(alpha); acc_phi.append(phi); acc_ent.append(ent)
        delta = alpha - phi
        for nome, mask in [("tutti", np.ones_like(y, bool)),
                           ("bona fide", y == 1), ("spoof", y == 0)]:
            righe.append({"variante": V, "seme": s, "sottoinsieme": nome,
                          "massa_silenzio": alpha[mask].mean(),
                          "frazione_silenzio": phi[mask].mean(),
                          "eccesso_delta": delta[mask].mean(),
                          "entropia_norm": np.nanmean(ent[mask])})
    if acc_alpha:
        per_variante[V] = (np.mean(acc_alpha, 0), np.mean(acc_phi, 0), np.mean(acc_ent, 0))

df = pd.DataFrame(righe)
df.to_csv(OUT_DIR / "attenzione_per_seme.csv", index=False)

sintesi = (df.groupby(["variante", "sottoinsieme"])
             .agg(massa_silenzio=("massa_silenzio", "mean"),
                  frazione_silenzio=("frazione_silenzio", "mean"),
                  eccesso_medio=("eccesso_delta", "mean"),
                  eccesso_std=("eccesso_delta", "std"),
                  entropia_norm=("entropia_norm", "mean"))
             .round(4))
sintesi.to_csv(OUT_DIR / "attenzione_sintesi.csv")
print("\n=== sintesi (media sui semi disponibili) ===")
print(sintesi)

# intervalli di confidenza sullo scarto delta, mediando prima sui semi
print("\n=== IC 95% dello scarto delta (media sui semi, bootstrap sui file) ===")
for V, (alpha, phi, ent) in per_variante.items():
    delta = alpha - phi
    for nome, mask in [("tutti", np.ones_like(y, bool)),
                       ("bona fide", y == 1), ("spoof", y == 0)]:
        lo, hi = ic95(delta[mask])
        print(f"{V} {nome:10s} delta medio {delta[mask].mean():+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]"
              f"{'   (esclude lo zero)' if lo > 0 or hi < 0 else ''}")

# ---- 3. figure ------------------------------------------------------------
if per_variante:
    V0 = "D" if "D" in per_variante else list(per_variante)[0]
    alpha, phi, ent = per_variante[V0]
    delta = alpha - phi

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for nome, mask, c in [("bona fide", y == 1, "C0"), ("spoof", y == 0, "C1")]:
        ax[0].hist(delta[mask], bins=60, alpha=0.6, density=True, label=nome, color=c)
    ax[0].axvline(0, color="k", ls=":", lw=1)
    ax[0].set_xlabel(r"$\delta_i$ = massa sul silenzio $-$ frazione di silenzio")
    ax[0].set_ylabel("densita'")
    ax[0].set_title(f"Variante {V0}: eccesso di attenzione sul silenzio")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    for nome, mask, c in [("bona fide", y == 1, "C0"), ("spoof", y == 0, "C1")]:
        ax[1].hist(ent[mask], bins=60, alpha=0.6, density=True, label=nome, color=c)
    ax[1].set_xlabel("entropia normalizzata dei pesi di attenzione")
    ax[1].set_ylabel("densita'")
    ax[1].set_title("1 = pesi uniformi, 0 = tutta la massa su un passo")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / "attenzione_silenzio.png", dpi=150)

    # profilo medio dell'attenzione sulle prime posizioni, normalizzato rispetto all'uniforme.
    # Si usa un solo seme: il profilo serve a mostrare l'andamento, non a stimare una media.
    K = 20
    primo_seme = next(s for s in SEMI if (ATT_DIR / f"attention_{V0}_seed{s}.npz").exists())
    att0 = carica_attenzione(ATT_DIR / f"attention_{V0}_seed{primo_seme}.npz")
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for nome, mask, c in [("bona fide", y == 1, "C0"), ("spoof", y == 0, "C1")]:
        M = np.full((int(mask.sum()), K), np.nan)
        for r, i in enumerate(np.where(mask)[0]):
            a = np.asarray(att0[i], dtype=np.float64)
            k = min(K, len(a))
            M[r, :k] = a[:k] * len(a)          # 1 = peso uniforme
        ax2.plot(np.arange(1, K + 1), np.nanmean(M, 0), marker=".", color=c, label=nome)
    ax2.axhline(1.0, color="k", ls=":", lw=1, label="peso uniforme")
    ax2.set_xlabel("posizione della finestra (1 = inizio del file)")
    ax2.set_ylabel("peso medio, normalizzato rispetto all'uniforme")
    ax2.set_title(f"Variante {V0}, seme {primo_seme}: attenzione sulle prime finestre")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / "attenzione_profilo.png", dpi=150)

print(f"\nFile scritti in {OUT_DIR.resolve()}")