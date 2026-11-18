"""Gerçek zamanlı Türkçe chat moderasyonu — Gradio arayüzü.

Çalıştırma:  python app.py                (modeller ./models'ta yoksa Google Drive'dan indirir)
             python app.py --models DIR   (modeller yerel klasördeyse)
             python app.py --share        (Colab'da herkese açık link verir)
"""
import argparse
import os

import gradio as gr
import pandas as pd

from src.models import load_all, timed_predict

# Model ağırlıkları repoda değil; herkese açık Google Drive klasöründe duruyor.
MODELS_URL = os.environ.get('MODELS_URL', 'https://drive.google.com/drive/folders/1qXhEniBflZYqNKvOaRUH1Sx1tRrpFTU5')
LATENCY_BUDGET_MS = 50
LABELS = {0: 'temiz', 1: 'saldırgan'}


def get_models_dir(arg_dir):
    if os.path.isfile(os.path.join(arg_dir, 'w2v_ffn.pt')):
        return arg_dir
    import gdown
    print(f'Modeller indiriliyor: {MODELS_URL}')
    gdown.download_folder(MODELS_URL, output=arg_dir, quiet=False)
    return arg_dir


def mask(text):
    return ' '.join('█' * len(w) for w in text.split())


def build_ui(models):
    names = list(models)

    def on_send(message, model_name, history, log):
        message = (message or '').strip()
        if not message:
            return '', history, log, log, None
        label, conf, ms = timed_predict(models[model_name], message)
        shown = f'🚫 {mask(message)}' if label else message
        verdict = f"{'🔴' if label else '🟢'} **{LABELS[label]}** · güven {conf:.2f} · {ms:.1f} ms"
        history = history + [{'role': 'user', 'content': shown},
                             {'role': 'assistant', 'content': verdict}]
        log = log + [{'#': len(log) + 1, 'model': model_name, 'etiket': LABELS[label],
                      'güven': round(conf, 3), 'süre (ms)': round(ms, 1),
                      f'≤{LATENCY_BUDGET_MS} ms': '✓' if ms <= LATENCY_BUDGET_MS else '✗'}]
        compare = pd.DataFrame([
            {'model': n, 'etiket': LABELS[l], 'güven': round(c, 3), 'süre (ms)': round(t, 1)}
            for n in names for l, c, t in [timed_predict(models[n], message)]])
        return '', history, log, pd.DataFrame(log), compare

    with gr.Blocks(title='Chat Moderasyonu') as demo:
        gr.Markdown('## 🎮 Gerçek zamanlı Türkçe chat moderasyonu\n'
                    'Her mesaj anında **temiz / saldırgan** olarak etiketlenir; saldırgan mesajlar gizlenir. '
                    f'Varsayılan model: **BERTurk INT8 (quantized)**, CPU hedefi ≤ {LATENCY_BUDGET_MS} ms.')
        log_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                try:                       # Gradio 5: type='messages'; Gradio 6'da varsayılan bu
                    chat = gr.Chatbot(type='messages', height=420, label='Sohbet')
                except TypeError:
                    chat = gr.Chatbot(height=420, label='Sohbet')
                with gr.Row():
                    box = gr.Textbox(placeholder='Mesaj yaz ve Enter’a bas…', show_label=False, scale=5)
                    send = gr.Button('Gönder', variant='primary', scale=1)
            with gr.Column(scale=2):
                model_dd = gr.Dropdown(names, value=names[0], label='Model')
                log_df = gr.Dataframe(label='Yanıt süresi günlüğü', interactive=False)
                compare_df = gr.Dataframe(label='Son mesaj: tüm modeller yan yana', interactive=False)
        inputs = [box, model_dd, chat, log_state]
        outputs = [box, chat, log_state, log_df, compare_df]
        box.submit(on_send, inputs, outputs)
        send.click(on_send, inputs, outputs)
    return demo


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='models', help='model klasörü (yoksa Google Drive’dan indirilir)')
    ap.add_argument('--share', action='store_true', help='herkese açık Gradio linki (Colab için)')
    ap.add_argument('--fp32', action='store_true', help='karşılaştırma için FP32 BERTurk’ü de yükle')
    args = ap.parse_args()
    models = load_all(get_models_dir(args.models), include_fp32=args.fp32)
    build_ui(models).launch(share=args.share)
