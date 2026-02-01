import csv
import tempfile

import gradio as gr
from transformers import pipeline

MODEL_NAME = "WpnSta/lner-xlm-roberta"

ner_pipeline = pipeline(
    "token-classification",
    model=MODEL_NAME,
    aggregation_strategy="simple",
)

def build_csv(text, entities):
    """Create a CSV file mapping every word to its entity label or 'O'."""
    # Split text into words while tracking character positions
    words = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        words.append((text[i:j], i, j))
        i = j

    # For each word, find if it overlaps with an entity span
    rows = []
    for word, start, end in words:
        label = "O"
        for ent in entities:
            if start >= ent["start"] and end <= ent["end"]:
                label = ent["entity_group"]
                break
            if start < ent["end"] and end > ent["start"]:
                label = ent["entity_group"]
                break
        rows.append((word, label))

    # Write to a temp CSV file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.writer(tmp)
    writer.writerow(["Word", "Label"])
    writer.writerows(rows)
    tmp.close()
    return tmp.name


def run_ner(text: str):
    if not text or not text.strip():
        return gr.update(value={"text": "", "entities": []}, visible=False), gr.update(value=None, visible=False)

    entities = ner_pipeline(text)

    # Convert to HighlightedText format
    highlighted_entities = []
    for ent in entities:
        highlighted_entities.append({
            "entity": ent["entity_group"],
            "start": ent["start"],
            "end": ent["end"],
        })

    highlighted = {"text": text, "entities": highlighted_entities}
    csv_path = build_csv(text, entities)
    return gr.update(value=highlighted, visible=True), gr.update(value=csv_path, visible=True)


def process_file(file):
    if file is None:
        return gr.update(value={"text": "Please upload a .txt file.", "entities": []}, visible=True), gr.update(value=None, visible=False)
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return gr.update(value={"text": "The uploaded file is empty.", "entities": []}, visible=True), gr.update(value=None, visible=False)
    return run_ner(text)


COLOR_MAP = {
    "PER": "#4A90D9",   # Blue
    "GPE": "#9B59B6",   # Purple
    "LOC": "#D94A4A",   # Red
    "ORG": "#4AD97A",   # Green
    "FAC": "#D9A34A",   # Orange
    "VEH": "#1ABC9C",   # Teal
    "TIME": "#E74C8B",  # Pink
}


theme = gr.themes.Default(
    primary_hue="stone",
    secondary_hue="neutral",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Raleway"),
)

with gr.Blocks(title="NER Literary Texts", theme=theme) as demo:
    gr.Markdown("# Named Entity Recognition for Literary Texts")
    gr.Markdown("Detect persons, places, organizations, and more in **English**, **French**, and **Italian** text. The texts will be analysed using a fine-tuned XLM-RoBERTa model that was trained with literary texts in these languages dating from the 19th to the 20th century. For more technical information see https://github.com/WpnSta/CAS_Mod4_NER.")

    with gr.Tabs():
        with gr.Tab("Text Input"):
            text_input = gr.Textbox(label="Enter or paste your text", lines=5)
            gr.Examples(
                examples=[
                    ["Although they had but that moment left the school behind them, they were now in the busy thoroughfares of a city, where shadowy passengers passed and re-passed; where shadowy carts and coaches battled for the way, and all the strife and tumult of a real city were. It was made plain enough, by the dressing of the shops, that here, too, it was Christmas-time again; but it was evening, and the streets were lighted up. The Ghost stopped at a certain warehouse door, and asked Scrooge if he knew it."],
                    ["Les cabines roulantes, attelées d'un cheval, remontaient aussi; et sur les planches de la promenade, qui borde la plage d'un bout à l'autre, c'était maintenant une coulée continue, épaisse et lente, de foule élégante, formant deux courants contraires qui se coudoyaient et se mêlaient. Pierre, nerveux, exaspéré par ce frôlement, s'enfuit, s'enfonça dans la ville et s'arrêta pour déjeuner chez un simple marchand de vins, à l'entrée des champs."],
                    ["Detto fatto traversarono la città, e, usciti fuori delle mura, si fermarono in un campo solitario che, su per giù, somigliava a tutti gli altri campi. Pinocchio è derubato delle sue monete d'oro, e per gastigo si busca quattro mesi di prigione. Il burattino, ritornato in città, cominciò a contare i minuti a uno a uno: e quando gli parve che fosse l'ora, riprese subito la strada che menava al Campo dei miracoli."],
                ],
                example_labels=[
                    "English - A Christmas Carol, Charles Dickens",
                    "French - Pierre et Jean, Guy de Maupassant",
                    "Italian - Le avventure di Pinocchio, Carlo Collodi",
                ],
                inputs=text_input,
            )
            text_button = gr.Button("Analyze", variant="primary")
            text_output = gr.HighlightedText(label="Detected Entities", color_map=COLOR_MAP, show_legend=True, visible=False)
            text_csv = gr.File(label="Download CSV", visible=False)
            text_button.click(fn=run_ner, inputs=text_input, outputs=[text_output, text_csv])

        with gr.Tab("File Upload"):
            file_input = gr.File(label="Upload a .txt file", file_types=[".txt"], type="filepath")
            file_button = gr.Button("Analyze File", variant="primary")
            file_output = gr.HighlightedText(label="Detected Entities", color_map=COLOR_MAP, show_legend=True, visible=False)
            file_csv = gr.File(label="Download CSV", visible=False)
            file_button.click(fn=process_file, inputs=file_input, outputs=[file_output, file_csv])

    gr.Markdown("### Entity Types\n| Tag | Meaning |\n|---|---|\n| PER | Person | \n| GPE | Geo-political entity |\n| LOC | Location |\n| ORG | Organization |\n| FAC | Facility |\n| VEH | Vehicle |\n| TIME | Temporal expression |")

if __name__ == "__main__":
    demo.launch()