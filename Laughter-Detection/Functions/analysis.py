import pandas as pd
import matplotlib.pyplot as plt

def analizar_resumen_risa(resumen_data, output_csv="resumen_risa.csv"):
    df_resumen = pd.DataFrame(resumen_data)
    if df_resumen.empty:
        print("⚠️ El resumen está vacío. Nada que analizar.")
        return
    df_resumen["porcentaje_risa"] = df_resumen["n_risa"] / df_resumen["n_segmentos"]
    df_resumen.to_csv(output_csv, index=False)
    print(f"✅ Resumen guardado como {output_csv}")
    conteo = df_resumen.groupby(["video", "pid"]).size().reset_index(name="n_repeticiones")
    conteo_duplicados = conteo[conteo["n_repeticiones"] > 1]
    if not conteo_duplicados.empty:
        print("\n🔁 Combinaciones repetidas (video, pid):")
        print(conteo_duplicados)
    else:
        print("\n✅ No hay combinaciones repetidas (video, pid).")
    print("\n🏆 Top 10 videos con más etiquetas de risa:")
    print(df_resumen.groupby("video")["n_risa"].sum().sort_values(ascending=False).head(10))
    df_resumen_clean = df_resumen.drop_duplicates(subset=["video", "pid"])
    plt.figure(figsize=(7, 4))
    df_resumen_clean["porcentaje_risa"].hist(bins=30)
    plt.title("Distribution of Laughter Percentage per Participant-Video")
    plt.xlabel("Laughter Percentage (%)")
    plt.ylabel("Number of Participant-Video Combinations")
    plt.tight_layout()
    plt.show()