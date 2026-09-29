#!/usr/bin/env python3
"""
Generate a clean, publication-grade pipeline architecture diagram for Task 2 Report.
Produces both PDF (vector) and PNG (high-res raster) versions.
Optimized aspect ratio (16 x 11.2) for perfect embedding in academic LaTeX papers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_pipeline_diagram(output_pdf_path, output_png_path):
    fig, ax = plt.subplots(figsize=(16, 11.2), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11.2)
    ax.axis('off')

    # Color Palette - Professional Academic / Tech
    c_bg_main = '#FFFFFF'
    c_stage1_bg = '#F8FAFC'  # Slate tint
    c_stage1_border = '#CBD5E1'
    c_stage2_bg = '#F0FDF4'  # Emerald tint
    c_stage2_border = '#86EFAC'
    c_stage3_bg = '#F0F9FF'  # Sky tint
    c_stage3_border = '#7DD3FC'
    c_stage4_bg = '#FFFBEB'  # Amber tint
    c_stage4_border = '#FCD34D'
    c_stage5_bg = '#FEF2F2'  # Rose tint
    c_stage5_border = '#FCA5A5'

    # Text & Accent Colors
    c_title = '#0F172A'
    c_body = '#334155'
    c_accent_blue = '#1D4ED8'
    c_accent_teal = '#0F766E'
    c_accent_green = '#047857'
    c_accent_amber = '#B45309'
    c_accent_rose = '#B91C1C'
    c_arrow = '#475569'

    fig.patch.set_facecolor(c_bg_main)
    ax.set_facecolor(c_bg_main)

    def draw_rounded_rect(x, y, w, h, bg, border, lw=1.4, rounding=0.12, zorder=1):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=bg, edgecolor=border, linewidth=lw, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_stage_header(x, y, w, h, stage_num, stage_title, badge_color):
        badge_w = 1.7
        badge_h = 0.32
        draw_rounded_rect(x + 0.2, y + h - badge_h - 0.12, badge_w, badge_h, badge_color, badge_color, lw=0, rounding=0.08, zorder=2)
        ax.text(x + 0.2 + badge_w / 2, y + h - 0.12 - badge_h / 2, f"STAGE {stage_num}",
                fontsize=8.5, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=3)
        ax.text(x + 0.2 + badge_w + 0.25, y + h - 0.12 - badge_h / 2, stage_title,
                fontsize=9.2, fontweight='bold', color=c_title, ha='left', va='center', zorder=3)

    def draw_card(x, y, w, h, title, bullets, title_color='#1E3A8A', border_color='#CBD5E1', bg_color='#FFFFFF', zorder=2):
        draw_rounded_rect(x, y, w, h, bg_color, border_color, lw=1.1, rounding=0.10, zorder=zorder)
        ax.text(x + w / 2, y + h - 0.18, title,
                fontsize=9.0, fontweight='bold', color=title_color, ha='center', va='top', zorder=zorder+1)
        ax.text(x + w / 2, y + h - 0.44, bullets,
                fontsize=7.8, color=c_body, ha='center', va='top', linespacing=1.35, zorder=zorder+1)

    def draw_arrow(x1, y1, x2, y2, color=c_arrow, lw=1.5, ls='-', zorder=5):
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>,head_length=4.5,head_width=3.2',
            color=color, linewidth=lw, linestyle=ls, zorder=zorder
        )
        ax.add_patch(arrow)

    # -------------------------------------------------------------
    # STAGE 1: DATA INGESTION & TARGET FORMULATION
    # -------------------------------------------------------------
    s1_y = 9.35
    s1_h = 1.65
    draw_rounded_rect(0.5, s1_y, 15.0, s1_h, c_stage1_bg, c_stage1_border, lw=1.4)
    draw_stage_header(0.5, s1_y, 15.0, s1_h, "1", "DATA INGESTION & TARGET FORMULATION", '#475569')

    bullets_1a = (
        "• Bank Indonesia JISDOR Daily USD/IDR Fixing (10:00 WIB)\n"
        "• Aligned Financial Headlines (GDELT Indonesian queries & Kontan)\n"
        "• 5-Year Chronological Horizon: Sep 30, 2021 to Sep 01, 2026 (1,168 days)"
    )
    draw_card(0.8, s1_y + 0.12, 6.7, 1.10, "Task 1 Aligned Dataset (final_dataset.csv)", bullets_1a,
              title_color=c_accent_blue, border_color='#93C5FD')

    bullets_1b = (
        r"• Binary Target: $y_t \in \{0, 1\}$ (UP: $P_t > P_{t-1}$, DOWN: $P_t < P_{t-1}$)" + "\n"
        "• Market Return Lags: $t-1$ to $t-5$ (only fixings published strictly prior to day $t$)\n"
        "• Decision Instant: strictly at market open before 10:00 WIB fixing announcement"
    )
    draw_card(8.5, s1_y + 0.12, 6.7, 1.10, "Target Definition & Past-Only Lags", bullets_1b,
              title_color=c_accent_blue, border_color='#93C5FD')

    draw_arrow(7.55, s1_y + 0.67, 8.45, s1_y + 0.67, color=c_accent_blue, lw=1.8)
    draw_arrow(8.0, s1_y, 8.0, 9.10, color=c_accent_green, lw=1.8)

    # -------------------------------------------------------------
    # STAGE 2: CHRONOLOGICAL PARTITIONING
    # -------------------------------------------------------------
    s2_y = 7.30
    s2_h = 1.75
    draw_rounded_rect(0.5, s2_y, 15.0, s2_h, c_stage2_bg, c_stage2_border, lw=1.4)
    draw_stage_header(0.5, s2_y, 15.0, s2_h, "2", "CHRONOLOGICAL PARTITIONING (Strict Zero Look-Ahead Protocol)", '#059669')

    splits = [
        (0.8, "TRAIN SPLIT (70%)", 
         "• 817 Trading Days (2021-09-30 to 2025-02-24)\n"
         "• Class Ratio: 56.18% UP / 43.82% DOWN\n"
         "• Role: Feature space fitting & model training", '#047857', '#A7F3D0'),
        (5.7, "VALIDATION SPLIT (15%)", 
         "• 175 Trading Days (2025-02-25 to 2025-11-24)\n"
         "• Class Ratio: 53.14% UP / 46.86% DOWN\n"
         "• Role: Hyperparameter grid tuning via MCC", '#0F766E', '#99F6E4'),
        (10.6, "TEST SPLIT (15%)", 
         "• 176 Trading Days (2025-11-25 to 2026-09-01)\n"
         "• Class Ratio: 59.09% UP / 40.91% DOWN\n"
         "• Role: Strict single forward evaluation (unseen)", '#15803D', '#BBF7D0')
    ]

    for x_pos, title, bullets, col, border_col in splits:
        draw_card(x_pos, s2_y + 0.12, 4.6, 1.20, title, bullets,
                  title_color=col, border_color=border_col)

    draw_arrow(4.0, s2_y, 4.0, 7.05, color=c_accent_blue, lw=1.6)
    draw_arrow(12.0, s2_y, 12.0, 7.05, color=c_accent_teal, lw=1.6)

    # -------------------------------------------------------------
    # STAGE 3: PARALLEL FEATURE EXTRACTION
    # -------------------------------------------------------------
    s3_y = 4.65
    s3_h = 2.35
    draw_rounded_rect(0.5, s3_y, 15.0, s3_h, c_stage3_bg, c_stage3_border, lw=1.4)
    draw_stage_header(0.5, s3_y, 15.0, s3_h, "3", "PARALLEL FEATURE EXTRACTION (Zero-Leakage Information Barrier)", '#0284C7')

    bullets_3a = (
        "• Past Return Lags: lag_1, lag_2, lag_3, lag_4, lag_5 (daily percentage returns)\n"
        "• Rolling Volatility: 5-day & 20-day return rolling standard deviations\n"
        "• Rolling Momentum: 5-day & 20-day return moving averages\n"
        "• Market Structure: 5-day up-ratio & calendar gap to previous trading day"
    )
    draw_card(0.8, s3_y + 0.52, 6.7, 1.35, "Market Time-Series Features (8 dims)", bullets_3a,
              title_color=c_accent_blue, border_color='#BAE6FD')

    bullets_3b = (
        "• Headline Preprocessing: lowercase, regex cleanup, Indonesian financial stopwords\n"
        "• InSet Lexicon Sentiment (3 dims): daily mean polarity score, pos & neg shares\n"
        "• TF-IDF + TruncatedSVD (20 dims): Latent Semantic Analysis topic vectors\n"
        "• Volume & Context Metadata (4 dims): log article count, GDELT tone, weekend share"
    )
    draw_card(8.5, s3_y + 0.52, 6.7, 1.35, "Classical Indonesian NLP Features (27 dims)", bullets_3b,
              title_color=c_accent_teal, border_color='#99F6E4')

    draw_rounded_rect(0.8, s3_y + 0.10, 14.4, 0.34, '#FEF3C7', '#F59E0B', lw=1.0, rounding=0.06, zorder=2)
    ax.text(8.0, s3_y + 0.27,
            "★ STRICT LEAKAGE PREVENTION: TF-IDF Vocabulary & SVD Matrices fitted ONLY on Train split (Transform-only on Val & Test)",
            fontsize=7.8, fontweight='bold', color='#B45309', ha='center', va='center', zorder=3)

    draw_arrow(4.15, s3_y, 4.15, 4.40, color=c_arrow, lw=1.5)
    draw_arrow(6.5, s3_y, 7.8, 4.40, color=c_arrow, lw=1.5)
    draw_arrow(11.85, s3_y, 8.2, 4.40, color=c_arrow, lw=1.5)

    # -------------------------------------------------------------
    # STAGE 4: BASELINE MODEL FAMILIES & MULTI-MODAL FUSION
    # -------------------------------------------------------------
    s4_y = 2.45
    s4_h = 1.90
    draw_rounded_rect(0.5, s4_y, 15.0, s4_h, c_stage4_bg, c_stage4_border, lw=1.4)
    draw_stage_header(0.5, s4_y, 15.0, s4_h, "4", "BASELINE MODEL FAMILIES & MULTI-MODAL FUSION", '#D97706')

    models = [
        (0.8, "Market-Only Models (8 dims)",
         "• Regularized Logistic Regression (L2 penalty)\n"
         "• XGBoost Classifier (depth 2-3, subsample 0.8)\n"
         "• Benchmark: pure price momentum & volatility", c_accent_amber, '#FDE68A'),
        (5.7, "Combined Models (35 dims)",
         "• Regularized Logistic Regression (Market + NLP)\n"
         "• XGBoost Classifier (Market + NLP features)\n"
         "• Key Hypothesis: text sentiment provides orthogonal alpha", '#B45309', '#FCD34D'),
        (10.6, "Naive Reference Baselines",
         "• Majority Class: always predicts UP (MCC = 0.000)\n"
         "• Persistence / Random Walk: $\hat{y}_t = y_{t-1}$\n"
         "• Purpose: rigorous sanity check against class skew", '#92400E', '#FDE68A')
    ]

    for x_pos, title, bullets, col, border_col in models:
        draw_card(x_pos, s4_y + 0.12, 4.6, 1.35, title, bullets,
                  title_color=col, border_color=border_col)

    draw_arrow(3.1, s4_y, 3.1, 2.20, color=c_arrow, lw=1.5)
    draw_arrow(8.0, s4_y, 8.0, 2.20, color=c_arrow, lw=1.5)
    draw_arrow(12.9, s4_y, 12.9, 2.20, color=c_arrow, lw=1.5)

    # -------------------------------------------------------------
    # STAGE 5: UNIFIED MODEL SELECTION & EVALUATION PROTOCOL
    # -------------------------------------------------------------
    s5_y = 0.25
    s5_h = 1.90
    draw_rounded_rect(0.5, s5_y, 15.0, s5_h, c_stage5_bg, c_stage5_border, lw=1.4)
    draw_stage_header(0.5, s5_y, 15.0, s5_h, "5", "UNIFIED MODEL SELECTION & EVALUATION PROTOCOL", '#DC2626')

    eval_steps = [
        (0.8, "Step 1: Validation Tuning",
         "• Grid search over hyperparameter space\n"
         "• Model trained on Train split (817 days)\n"
         "• Scored on Validation split (175 days)\n"
         "• Criterion: maximize MCC (robust to UP skew)", '#B91C1C', '#FECACA'),
        (5.7, "Step 2: Model Refitting",
         "• Lock selected optimal hyperparameters\n"
         "• Refit model on Train + Validation (992 days)\n"
         "• Refit TF-IDF & SVD on Train + Val text\n"
         "• Exploits maximum historical data for testing", '#991B1B', '#FCA5A5'),
        (10.6, "Step 3: Unbiased Test Evaluation",
         "• Single forward evaluation on Test (176 days)\n"
         "• Primary metrics: MCC & Macro-F1\n"
         "• Secondary metrics: Directional Accuracy & ROC-AUC\n"
         "• True out-of-sample real-world simulation", '#7F1D1D', '#FECACA')
    ]

    for x_pos, title, bullets, col, border_col in eval_steps:
        draw_card(x_pos, s5_y + 0.12, 4.6, 1.35, title, bullets,
                  title_color=col, border_color=border_col)

    draw_arrow(5.45, s5_y + 0.79, 5.65, s5_y + 0.79, color=c_accent_rose, lw=1.8)
    draw_arrow(10.35, s5_y + 0.79, 10.55, s5_y + 0.79, color=c_accent_rose, lw=1.8)

    plt.tight_layout(pad=0.15)
    
    plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Successfully generated:\n- {output_pdf_path}\n- {output_png_path}")

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    figures_dir = base_dir / 'report' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = figures_dir / 'pipeline_diagram.pdf'
    png_out = figures_dir / 'pipeline_diagram.png'
    create_pipeline_diagram(pdf_out, png_out)
