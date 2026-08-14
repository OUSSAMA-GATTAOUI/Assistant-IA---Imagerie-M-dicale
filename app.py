import os
import time
from datetime import datetime
from io import BytesIO
import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage)
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
MODELS_DIR = "models"
POUMONS_MODEL_PATH = os.path.join(MODELS_DIR, "poumons_classification_model.keras")
CERVEAU_MODEL_PATH = os.path.join(MODELS_DIR, "cerveau_classification_model.keras")
TYPE_IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "type_image_classification_model.keras")
TYPE_IMAGE_CLASSES_PATH = os.path.join(MODELS_DIR, "type_image_classes.txt")
IMG_DIMS = 224
SEUIL_CONFIANCE_TYPE_IMAGE = 0.70 
CLASSES_POUMONS = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]
CLASSES_CERVEAU = ["glioma", "meningioma", "notumor", "pituitary"]
LIBELLES_POUMONS = {
    "COVID": (
        "COVID-19",
        "Opacités bilatérales à prédominance périphérique et basale, aspect en verre dépoli.",
        "Aspect radiologique compatible avec une pneumonie d'origine virale (COVID-19). "
        "Corrélation clinique et biologique recommandée.",
    ),
    "Lung Opacity": (
        "Opacité pulmonaire",
        "Présence d'une ou plusieurs opacités pulmonaires non systématisées.",
        "Aspect compatible avec une opacité pulmonaire non spécifique. "
        "Corrélation clinique recommandée.",
    ),
    "Normal": (
        "Normal",
        "Transparence pulmonaire conservée des deux côtés. Absence d'opacité "
        "parenchymateuse ou d'épanchement pleural. Silhouette cardiomédiastinale "
        "de taille normale.",
        "Radiographie thoracique sans anomalie décelable.",
    ),
    "Viral Pneumonia": (
        "Pneumonie virale",
        "Opacité alvéolaire avec possible bronchogramme aérien, évoquant un foyer infectieux.",
        "Aspect compatible avec un foyer de pneumonie virale. Corrélation clinique recommandée.",
    ),
}
LIBELLES_CERVEAU = {
    "glioma": (
        "Gliome",
        "Masse intra-axiale de signal hétérogène, avec possible effet de masse sur "
        "les structures adjacentes.",
        "Aspect évocateur d'un gliome. Corrélation clinique et discussion en réunion "
        "de concertation pluridisciplinaire recommandées.",
    ),
    "meningioma": (
        "Méningiome",
        "Masse extra-axiale bien circonscrite, à large base d'implantation durale.",
        "Aspect évocateur d'un méningiome. Corrélation clinique recommandée.",
    ),
    "notumor": (
        "Pas de tumeur",
        "Parenchyme cérébral d'aspect normal. Absence de masse ou de lésion focale visible.",
        "IRM cérébrale sans anomalie décelable.",
    ),
    "pituitary": (
        "Tumeur hypophysaire",
        "Anomalie de signal au niveau de la région hypophysaire.",
        "Aspect évocateur d'une atteinte hypophysaire. Corrélation clinique et "
        "endocrinienne recommandée.",
    ),
}
TECHNIQUE_PAR_MODALITE = {
    "Radiographie thoracique": "Radiographie thoracique de face, incidence postéro-antérieure.",
    "IRM cérébrale": "IRM cérébrale, séquences pondérées T1/T2.",
}
CLASSE_NORMALE_PAR_MODALITE = {
    "Radiographie thoracique": "Normal",
    "IRM cérébrale": "notumor",
}
st.set_page_config(page_title="Assistant IA — Imagerie médicale", page_icon="🩺", layout="wide")
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }

    :root {
        --accent: #2563eb;
        --ink: #1f2937;
        --muted: #6b7280;
        --line: #e5e7eb;
        --ok: #16a34a;
        --warn: #d97706;
        --bad: #dc2626;
    }

    .card {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        background: #ffffff;
        margin-bottom: 1rem;
    }
    .card h4 { margin-top: 0; color: var(--ink); font-size: 0.95rem;
               text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }

    .report-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.55rem 0; border-bottom: 1px solid var(--line);
    }
    .report-row:last-child { border-bottom: none; }
    .report-label { color: var(--muted); font-size: 0.92rem; }
    .report-value { color: var(--ink); font-weight: 600; font-size: 0.98rem; text-align: right; }

    .badge {
        display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
        font-size: 0.82rem; font-weight: 600;
    }
    .badge-ok    { background: #eafaf0; color: var(--ok); }
    .badge-warn  { background: #fef6e7; color: var(--warn); }
    .badge-bad   { background: #fdecec; color: var(--bad); }

    .disclaimer {
        font-size: 0.82rem; color: var(--muted); border-top: 1px solid var(--line);
        margin-top: 1.5rem; padding-top: 0.75rem;
    }

    .rapport-entete {
        display: flex; justify-content: space-between; flex-wrap: wrap;
        gap: 0.5rem 2rem; padding-bottom: 0.9rem; margin-bottom: 0.9rem;
        border-bottom: 1px solid var(--line);
    }
    .rapport-entete .champ { font-size: 0.9rem; }
    .rapport-entete .champ .label { color: var(--muted); }
    .rapport-entete .champ .valeur { color: var(--ink); font-weight: 600; }

    .rapport-section { margin-bottom: 0.9rem; }
    .rapport-section:last-of-type { margin-bottom: 0; }
    .rapport-section .titre {
        color: var(--muted); font-size: 0.78rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;
    }
    .rapport-section .texte { color: var(--ink); font-size: 0.95rem; line-height: 1.5; }
    .rapport-section .texte.conclusion { font-weight: 600; }

    /* Résultat de l'analyse */
    .result-box {
        background: #eafaf0; border-radius: 8px; padding: 0.9rem 1rem;
        display: flex; align-items: flex-start; gap: 0.65rem; margin-bottom: 1.1rem;
    }
    .result-check {
        width: 1.9rem; height: 1.9rem; border-radius: 50%; background: var(--ok);
        color: #fff; display: flex; align-items: center; justify-content: center;
        font-size: 0.95rem; flex-shrink: 0;
    }
    .result-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin-bottom: 0.3rem; }
    .result-conf-label { color: var(--muted); font-size: 0.78rem; }
    .result-conf-value { font-size: 1.25rem; font-weight: 700; color: var(--ink); margin-top: 0.05rem; }

    .prob-section-title {
        color: var(--muted); font-size: 0.78rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em; margin: 0.2rem 0 0.7rem 0;
    }
    .prob-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.5rem 0; }
    .prob-name {
        width: 40%; font-size: 0.86rem; color: var(--ink); flex-shrink: 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .prob-track { flex: 1; height: 6px; background: #eef0f4; border-radius: 999px; overflow: hidden; }
    .prob-fill { height: 100%; background: var(--accent); border-radius: 999px; }
    .prob-pct { width: 50px; text-align: right; font-size: 0.82rem; color: var(--ink); font-weight: 600; }

    /* Informations sur l'analyse */
    .info-row {
        display: flex; align-items: center; gap: 0.65rem; padding: 0.55rem 0;
        border-bottom: 1px solid var(--line);
    }
    .info-row:last-child { border-bottom: none; }
    .info-icon {
        width: 1.8rem; height: 1.8rem; border-radius: 6px; background: #eef2ff;
        color: var(--accent); display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; flex-shrink: 0;
    }
    .info-label { color: var(--muted); font-size: 0.88rem; flex: 1; }
    .info-value { color: var(--ink); font-weight: 600; font-size: 0.88rem; text-align: right; }

    /* Analyses récentes */
    .hist-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .hist-table th {
        text-align: left; color: var(--muted); font-weight: 700; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.03em; padding: 0.4rem 0.5rem;
        border-bottom: 1px solid var(--line);
    }
    .hist-table td { padding: 0.5rem; border-bottom: 1px solid var(--line); vertical-align: middle; }
    .hist-table tr:last-child td { border-bottom: none; }
    .hist-thumb { width: 32px; height: 32px; border-radius: 6px; object-fit: cover; background: #eee; }
    .hist-empty { color: var(--muted); font-size: 0.88rem; padding: 0.4rem 0; }

    /* Statistiques rapides */
    .stat-card {
        border: 1px solid var(--line); border-radius: 10px; padding: 0.9rem 1rem;
        background: #ffffff; height: 100%;
    }
    .stat-icon {
        width: 2rem; height: 2rem; border-radius: 8px; display: flex;
        align-items: center; justify-content: center; font-size: 0.95rem; margin-bottom: 0.6rem;
    }
    .stat-icon-accent { background: #eef2ff; color: var(--accent); }
    .stat-icon-bad    { background: #fdecec; color: var(--bad); }
    .stat-icon-ok     { background: #eafaf0; color: var(--ok); }
    .stat-value { font-size: 1.3rem; font-weight: 700; color: var(--ink); }
    .stat-label { font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)
@st.cache_resource(show_spinner=False)
def charger_modeles():
    modele_type_image = load_model(TYPE_IMAGE_MODEL_PATH)
    modele_poumons = load_model(POUMONS_MODEL_PATH)
    modele_cerveau = load_model(CERVEAU_MODEL_PATH)

    with open(TYPE_IMAGE_CLASSES_PATH) as f:
        classes_type_image = [l.strip() for l in f if l.strip()]
    return modele_type_image, modele_poumons, modele_cerveau, classes_type_image
def preparer_image(image_pil, taille=IMG_DIMS):
    image_pil = image_pil.convert("RGB").resize((taille, taille))
    tableau = np.array(image_pil).astype("float32")
    tableau_pretraite = preprocess_input(tableau.copy())
    return image_pil, tableau_pretraite.reshape(1, taille, taille, 3)
def generer_gradcam(model, img_array, nom_derniere_couche_conv="relu"):
    base_model = model.layers[0]
    derniere_couche_conv = base_model.get_layer(nom_derniere_couche_conv)
    modele_conv = tf.keras.models.Model(base_model.input, derniere_couche_conv.output)
    with tf.GradientTape() as tape:
        sortie_conv = modele_conv(img_array, training=False)
        tape.watch(sortie_conv)
        x = sortie_conv
        for couche in model.layers[1:]:
            x = couche(x, training=False)
        indice_classe = tf.argmax(x[0])
        score_classe = x[:, indice_classe]
    gradients = tape.gradient(score_classe, sortie_conv)
    gradients_moyennes = tf.reduce_mean(gradients, axis=(0, 1, 2))
    sortie_conv = sortie_conv[0]
    carte_chaleur = sortie_conv @ gradients_moyennes[..., tf.newaxis]
    carte_chaleur = tf.squeeze(carte_chaleur)
    carte_chaleur = tf.maximum(carte_chaleur, 0) / (tf.math.reduce_max(carte_chaleur) + 1e-8)
    return carte_chaleur.numpy()
def superposer_chaleur(image_pil, carte_chaleur, taille=IMG_DIMS):
    carte_redim = np.array(Image.fromarray(carte_chaleur).resize((taille, taille)))
    colormap = matplotlib.colormaps["jet"]
    carte_couleur = (colormap(carte_redim)[:, :, :3] * 255).astype("uint8")
    return (0.6 * np.array(image_pil) + 0.4 * carte_couleur).astype("uint8")


def controle_qualite(image_pil):
    alertes = []
    largeur, hauteur = image_pil.size
    if largeur < 150 or hauteur < 150:
        alertes.append("Résolution faible : la fiabilité de la prédiction peut être réduite.")
    gris = np.array(image_pil.convert("L"), dtype="float32")
    nettete = np.var(np.gradient(gris))
    if nettete < 8:
        alertes.append("Image potentiellement floue : la fiabilité de la prédiction peut être réduite.")

    return alertes


def afficher_carte_html(html):
    lignes = [ligne.strip() for ligne in html.strip("\n").split("\n")]
    st.markdown("\n".join(lignes), unsafe_allow_html=True)


def generer_pdf_rapport(nom_affiche, date_affichee, nom_modalite, renseignements_affiches,
                         technique_texte, libelle_fr, resultats_texte, conclusion_texte,
                         confiance_expert, niveau, diagnostic_differentiel, image_superposition):
    tampon = BytesIO()
    doc = SimpleDocTemplate(tampon, pagesize=A4,
                             topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    styles = getSampleStyleSheet()
    style_titre = ParagraphStyle("TitreRapport", parent=styles["Title"], fontSize=15)
    style_section = ParagraphStyle("Section", parent=styles["Heading3"],
                                    fontSize=10, textColor=colors.HexColor("#6b7280"),
                                    spaceBefore=10, spaceAfter=2)
    style_texte = ParagraphStyle("Texte", parent=styles["Normal"], fontSize=10.5, leading=15)
    style_conclusion = ParagraphStyle("Conclusion", parent=style_texte, fontName="Helvetica-Bold")
    style_disclaimer = ParagraphStyle("Disclaimer", parent=styles["Normal"],
                                       fontSize=8, textColor=colors.HexColor("#6b7280"))

    elements = [
        Paragraph("Compte-rendu d'analyse — Assistant IA Imagerie Médicale", style_titre),
        Spacer(1, 10),
    ]

    entete = Table(
        [["Patient", nom_affiche, "Date de l'examen", date_affichee],
         ["Type d'image", nom_modalite, "Rapport généré le", datetime.now().strftime("%d/%m/%Y %H:%M")]],
        colWidths=[3 * cm, 4.5 * cm, 3.3 * cm, 4.2 * cm],
    )
    entete.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b7280")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#6b7280")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.75, colors.HexColor("#e5e7eb")),
    ]))
    elements += [entete, Spacer(1, 6)]

    sections = [
        ("Renseignements cliniques", renseignements_affiches, style_texte),
        ("Technique", technique_texte, style_texte),
        ("Type de maladie détectée", libelle_fr, style_conclusion),
        ("Résultats", resultats_texte, style_texte),
        ("Conclusion", conclusion_texte, style_conclusion),
    ]
    for titre, contenu, style in sections:
        elements.append(Paragraph(titre.upper(), style_section))
        elements.append(Paragraph(contenu, style))

    elements.append(Paragraph("DIAGNOSTIC DIFFÉRENTIEL", style_section))
    lignes_diff = [["Hypothèse", "Probabilité"]] + [
        [nom, f"{proba:.0%}"] for nom, proba in diagnostic_differentiel
    ]
    table_diff = Table(lignes_diff, colWidths=[9 * cm, 4 * cm])
    table_diff.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#e5e7eb")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements += [table_diff, Spacer(1, 6)]

    elements.append(Paragraph("CONFIANCE DU MODÈLE IA", style_section))
    elements.append(Paragraph(f"{confiance_expert:.0%} — Niveau {niveau}", style_texte))

    if image_superposition is not None:
        tampon_image = BytesIO()
        Image.fromarray(image_superposition).save(tampon_image, format="PNG")
        tampon_image.seek(0)
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("ZONE ANALYSÉE (GRAD-CAM)", style_section))
        elements.append(RLImage(tampon_image, width=7 * cm, height=7 * cm))

    elements.append(Spacer(1, 16))
    elements.append(Paragraph(
        "Ce rapport est généré par un outil d'aide à la décision basé sur l'intelligence "
        "artificielle. Il ne remplace en aucun cas l'avis d'un professionnel de santé.",
        style_disclaimer,
    ))

    doc.build(elements)
    tampon.seek(0)
    return tampon.getvalue()


def niveau_confiance(confiance):
    if confiance >= 0.90:
        return "Élevé", "badge-ok"
    elif confiance >= 0.70:
        return "Moyen", "badge-warn"
    return "Faible", "badge-bad"


def image_vers_base64_thumb(image_pil, taille=64):
    import base64
    miniature = image_pil.convert("RGB").copy()
    miniature.thumbnail((taille, taille))
    tampon = BytesIO()
    miniature.save(tampon, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(tampon.getvalue()).decode()


def enregistrer_historique(image_originale, libelle_fr, confiance_expert, est_normal, cle_fichier):
    if "historique" not in st.session_state:
        st.session_state.historique = []
    deja_present = any(h["fichier_id"] == cle_fichier for h in st.session_state.historique)
    if not deja_present:
        st.session_state.historique.append({
            "fichier_id": cle_fichier,
            "miniature": image_vers_base64_thumb(image_originale),
            "libelle": libelle_fr,
            "confiance": confiance_expert,
            "date": datetime.now().strftime("%d/%m %H:%M"),
            "est_normal": est_normal,
        })


def calculer_statistiques():
    historique = st.session_state.get("historique", [])
    total = len(historique)
    anomalies = sum(1 for h in historique if not h["est_normal"])
    normaux = total - anomalies
    confiance_moyenne = (sum(h["confiance"] for h in historique) / total) if total else 0.0
    return total, anomalies, normaux, confiance_moyenne


def afficher_analyses_recentes(limite=5):
    historique = list(reversed(st.session_state.get("historique", [])))[:limite]

    afficher_carte_html('<div class="card"><h4>Analyses récentes</h4>')
    if not historique:
        afficher_carte_html('<p class="hist-empty">Aucune analyse pour le moment.</p>')
    else:
        lignes = ""
        for h in historique:
            couleur = "var(--bad)" if not h["est_normal"] else "var(--ok)"
            lignes += f"""
            <tr>
                <td><img class="hist-thumb" src="{h['miniature']}"></td>
                <td style="color:{couleur}; font-weight:600;">{h['libelle']}</td>
                <td>{h['confiance']:.1%}</td>
                <td>{h['date']}</td>
            </tr>
            """
        afficher_carte_html(f"""
        <table class="hist-table">
            <thead><tr><th>Image</th><th>Résultat</th><th>Conf.</th><th>Date</th></tr></thead>
            <tbody>{lignes}</tbody>
        </table>
        """)
    afficher_carte_html("</div>")


def afficher_statistiques_rapides():
    total, anomalies, normaux, confiance_moyenne = calculer_statistiques()
    donnees = [
        ("Images analysées", str(total)),
        ("Anomalies détectées", str(anomalies)),
        ("Images normales", str(normaux)),
        ("Confiance moyenne", f"{confiance_moyenne:.1%}" if total else "—"),
    ]
    lignes = ""
    for label, valeur in donnees:
        lignes += f"""
        <div class="info-row">
            <div class="info-label">{label}</div>
            <div class="info-value">{valeur}</div>
        </div>
        """
    afficher_carte_html(f"""
    <div class="card">
        <h4>Statistiques rapides</h4>
        {lignes}
    </div>
    """)
st.markdown("## Assistant IA — Imagerie médicale")
st.markdown(
    "<span style='color:#6b7280;'>Dépose une radiographie thoracique ou une IRM cérébrale. "
    "Le système reconnaît le type d'image, l'envoie au modèle spécialiste, "
    "et affiche une fiche de résultat expliquée par Grad-CAM.</span>",
    unsafe_allow_html=True,
)
st.write("")

col_nom, col_date = st.columns(2)
with col_nom:
    nom_patient = st.text_input("Nom du patient (optionnel)", value="")
with col_date:
    date_examen = st.date_input("Date de l'examen", value=datetime.now())

renseignements_cliniques = st.text_input(
    "Renseignements cliniques (optionnel)",
    value="",
    placeholder="Ex. : toux persistante, bilan avant chirurgie...",
)

fichier = st.file_uploader("Déposer une image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if fichier is not None:
    image_originale = Image.open(fichier)

    with st.spinner("Chargement des modèles..."):
        modele_type_image, modele_poumons, modele_cerveau, classes_type_image = charger_modeles()

    col_image, col_type = st.columns([1, 1.3])

    with col_image:
        st.image(image_originale, caption="Image envoyée", use_container_width=True)
        alertes_qualite = controle_qualite(image_originale)
        for alerte in alertes_qualite:
            st.markdown(f"<div class='badge badge-warn'>⚠ {alerte}</div>", unsafe_allow_html=True)

    with st.spinner("Analyse en cours..."):
        image_pil, tableau = preparer_image(image_originale)

        # Étape 1 — type d'image
        probabilites_type = modele_type_image.predict(tableau, verbose=0)[0]
        indice_type = int(np.argmax(probabilites_type))
        type_predit = classes_type_image[indice_type]
        confiance_type = float(probabilites_type[indice_type])

        incertain = confiance_type < SEUIL_CONFIANCE_TYPE_IMAGE
        non_supporte = (type_predit == "Autre") or incertain

    if non_supporte:
        with col_type:
            afficher_carte_html("""
            <div class="card">
                <h4>Type d'image</h4>
                <span class="badge badge-bad">Non reconnu</span>
                <p style="color:#1f2937; margin-top:0.75rem;">
                    Cette image ne correspond ni à une radiographie thoracique ni à une IRM
                    cérébrale avec suffisamment de certitude. Merci d'envoyer l'un de ces
                    deux types d'image.
                </p>
            </div>
            """)
        st.stop()

    if True:
        if type_predit == "Radiographie":
            modele_expert = modele_poumons
            classes_expert = CLASSES_POUMONS
            libelles_expert = LIBELLES_POUMONS
            nom_modalite = "Radiographie thoracique"
        elif type_predit == "IRM":
            modele_expert = modele_cerveau
            classes_expert = CLASSES_CERVEAU
            libelles_expert = LIBELLES_CERVEAU
            nom_modalite = "IRM cérébrale"
        else:
            st.error("Type d'image inattendu — impossible de router vers un modèle spécialiste.")
            st.stop()

        debut_chrono = time.time()
        probabilites_expert = modele_expert.predict(tableau, verbose=0)[0]
        indice_expert = int(np.argmax(probabilites_expert))
        classe_predite = classes_expert[indice_expert]
        confiance_expert = float(probabilites_expert[indice_expert])
        libelle_fr, resultats_texte, conclusion_texte = libelles_expert[classe_predite]
        niveau, badge_classe = niveau_confiance(confiance_expert)
        technique_texte = TECHNIQUE_PAR_MODALITE[nom_modalite]
        diagnostic_differentiel = sorted(
            [(libelles_expert[c][0], float(p)) for c, p in zip(classes_expert, probabilites_expert)],
            key=lambda t: t[1], reverse=True,
        )

        temps_analyse = time.time() - debut_chrono
        nom_modele_utilise = "DenseNet121 (poumons)" if nom_modalite == "Radiographie thoracique" else "DenseNet121 (cerveau)"

        carte_chaleur = generer_gradcam(modele_expert, tableau)
        superposition = superposer_chaleur(image_pil, carte_chaleur)

        lignes_proba = ""
        for nom, proba in diagnostic_differentiel:
            lignes_proba += f"""
            <div class="prob-row">
                <div class="prob-name">{nom}</div>
                <div class="prob-track"><div class="prob-fill" style="width:{proba*100:.1f}%;"></div></div>
                <div class="prob-pct">{proba:.1%}</div>
            </div>
            """

        col_gradcam, col_resultat = st.columns([1, 1.3])
        with col_gradcam:
            st.image(superposition, caption="Zone analysée par le modèle (Grad-CAM)", use_container_width=True)
        with col_resultat:
            afficher_carte_html(f"""
            <div class="card">
                <h4>Résultat de l'analyse</h4>
                <div class="result-box">
                    <div class="result-check">✓</div>
                    <div>
                        <div class="result-title">{libelle_fr}</div>
                        <div class="result-conf-label">Confiance du modèle</div>
                        <div class="result-conf-value">{confiance_expert:.1%}</div>
                    </div>
                </div>
                <div class="prob-section-title">Probabilités par classe</div>
                {lignes_proba}
            </div>
            """)
        cle_fichier = getattr(fichier, "file_id", fichier.name + str(fichier.size))
        est_normal = classe_predite == CLASSE_NORMALE_PAR_MODALITE.get(nom_modalite)
        enregistrer_historique(image_originale, libelle_fr, confiance_expert, est_normal, cle_fichier)

        col_infos, col_hist, col_stats = st.columns([0.9, 1.2, 0.9])
        with col_infos:
            afficher_carte_html(f"""
            <div class="card">
                <h4>Informations sur l'analyse</h4>
                <div class="info-row">
                    <div class="info-label">Date</div>
                    <div class="info-value">{datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">Temps d'analyse</div>
                    <div class="info-value">{temps_analyse:.2f} seconde</div>
                </div>
                <div class="info-row">
                    <div class="info-label">Modèle utilisé</div>
                    <div class="info-value">{nom_modele_utilise}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">Type d'image</div>
                    <div class="info-value">{nom_modalite}</div>
                </div>
            </div>
            """)
        with col_hist:
            afficher_analyses_recentes(limite=5)
        with col_stats:
            afficher_statistiques_rapides()

        nom_affiche = nom_patient.strip() if nom_patient.strip() else "Non renseigné"
        renseignements_affiches = (
            renseignements_cliniques.strip() if renseignements_cliniques.strip() else "Non renseigné"
        )
        date_affichee = date_examen.strftime("%d/%m/%Y")

        diagnostic_html = "<br>".join(
            f"{nom} — {proba:.0%}" for nom, proba in diagnostic_differentiel
        )

        afficher_carte_html(f"""
        <div class="card">
            <div class="rapport-entete">
                <div class="champ"><span class="label">Patient : </span>
                    <span class="valeur">{nom_affiche}</span></div>
                <div class="champ"><span class="label">Date de l'examen : </span>
                    <span class="valeur">{date_affichee}</span></div>
            </div>

            <div class="rapport-section">
                <div class="titre">Renseignements cliniques</div>
                <div class="texte">{renseignements_affiches}</div>
            </div>

            <div class="rapport-section">
                <div class="titre">Technique</div>
                <div class="texte">{technique_texte}</div>
            </div>

            <div class="rapport-section">
                <div class="titre">Type de maladie détectée</div>
                <div class="texte conclusion">{libelle_fr}</div>
            </div>

            <div class="rapport-section">
                <div class="titre">Diagnostic différentiel</div>
                <div class="texte">{diagnostic_html}</div>
            </div>

            <div class="rapport-section">
                <div class="titre">Résultats</div>
                <div class="texte">{resultats_texte}</div>
            </div>

            <div class="rapport-section">
                <div class="titre">Conclusion</div>
                <div class="texte conclusion">{conclusion_texte}</div>
            </div>

            <div class="rapport-section">
                <div class="titre">Confiance du modèle IA</div>
                <div class="texte">{confiance_expert:.0%}
                    <span class="badge {badge_classe}">{niveau}</span>
                </div>
            </div>
        </div>
        """)

        diagnostic_texte = "\n".join(
            f"  - {nom:<25s} {proba:.0%}" for nom, proba in diagnostic_differentiel
        )

        rapport_texte = (
            "COMPTE-RENDU DE RADIOGRAPHIE / IRM - ASSISTANT IA IMAGERIE MEDICALE\n"
            "----------------------------------------------------------------------\n"
            f"Patient          : {nom_affiche}\n"
            f"Date de l'examen : {date_affichee}\n"
            f"Type d'image     : {nom_modalite}\n"
            "----------------------------------------------------------------------\n"
            "TYPE DE MALADIE DETECTEE\n"
            f"{libelle_fr}\n\n"
            "DIAGNOSTIC DIFFERENTIEL\n"
            f"{diagnostic_texte}\n\n"
            "RENSEIGNEMENTS CLINIQUES\n"
            f"{renseignements_affiches}\n\n"
            "TECHNIQUE\n"
            f"{technique_texte}\n\n"
            "RESULTATS\n"
            f"{resultats_texte}\n\n"
            "CONCLUSION\n"
            f"{conclusion_texte}\n\n"
            f"Confiance du modele IA : {confiance_expert:.0%} ({niveau})\n"
            f"Rapport genere le      : {datetime.now().strftime('%d/%m/%Y a %H:%M')}\n"
            "----------------------------------------------------------------------\n"
            "Ce rapport est genere par un outil d'aide a la decision base sur l'IA.\n"
            "Il ne remplace pas un diagnostic medical professionnel.\n"
        )
        nom_fichier_rapport = f"rapport_{(nom_patient.strip() or 'patient').replace(' ', '_')}.txt"

        pdf_bytes = generer_pdf_rapport(
            nom_affiche, date_affichee, nom_modalite, renseignements_affiches,
            technique_texte, libelle_fr, resultats_texte, conclusion_texte,
            confiance_expert, niveau, diagnostic_differentiel, superposition,
        )
        nom_fichier_pdf = f"rapport_{(nom_patient.strip() or 'patient').replace(' ', '_')}.pdf"

        col_txt, col_pdf = st.columns(2)
        with col_txt:
            st.download_button("Télécharger le compte-rendu (.txt)", rapport_texte,
                                file_name=nom_fichier_rapport)
        with col_pdf:
            st.download_button("Télécharger le compte-rendu (.pdf)", pdf_bytes,
                                file_name=nom_fichier_pdf, mime="application/pdf")

        st.markdown(
            "<div class='disclaimer'>Cet outil est une aide à la décision basée sur "
            "l'intelligence artificielle. Il ne remplace en aucun cas l'avis d'un "
            "professionnel de santé.</div>",
            unsafe_allow_html=True,
        )