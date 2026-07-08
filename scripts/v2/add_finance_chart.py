import json
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Generate the Image ---
def generate_image():
    # Data
    categories = ['OPEX Loss\n(Storage)', 'P&L Loss\n(Missed Sales)', 'Total Financial\nLoss']
    baseline = [372995, 3729951, 4102946]
    ensemble = [121390, 3347175, 3468566]

    x = np.arange(len(categories))
    width = 0.35

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    rects1 = ax.bar(x - width/2, baseline, width, label='Baseline (Single LightGBM)', color='#444c56')
    rects2 = ax.bar(x + width/2, ensemble, width, label='Ultimate Meta-Ensemble', color='#8a2be2')

    ax.set_ylabel('Financial Loss ($ USD)', fontsize=12, fontweight='bold', color='#c9d1d9')
    ax.set_title('Business Impact: Financial Loss Comparison', fontsize=16, fontweight='bold', color='#ffffff', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='#c9d1d9')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=11, loc='upper left')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'${height/1000:.0f}K',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ffffff')

    autolabel(rects1)
    autolabel(rects2)

    # Highlight Savings
    ax.annotate('Net Savings:\n$634,380', xy=(2.17, 3468566), xytext=(2.6, 3800000),
                arrowprops=dict(facecolor='#3fb950', shrink=0.05, width=2, headwidth=8),
                fontsize=13, fontweight='bold', color='#3fb950', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='#161b22', ec='#3fb950', lw=2))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.tick_params(colors='#c9d1d9')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#c9d1d9')

    plt.tight_layout()
    plt.savefig('visualizations/08_Financial_Impact_ROI.png', facecolor=fig.get_facecolor(), edgecolor='none')
    print("Generated visualizations/08_Financial_Impact_ROI.png")

# --- 2. Update the Jupyter Notebook ---
def update_notebook():
    notebook_path = 'visualizations/m5_total_visualizations.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    # Markdown Cell
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Business Impact: Financial ROI Comparison\n",
            "Visualizing the massive OPEX and P&L savings ($634K) achieved by the Meta-Ensemble."
        ]
    }

    # Code Cell
    code_source = [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "# Data\n",
        "categories = ['OPEX Loss\\n(Storage)', 'P&L Loss\\n(Missed Sales)', 'Total Financial\\nLoss']\n",
        "baseline = [372995, 3729951, 4102946]\n",
        "ensemble = [121390, 3347175, 3468566]\n",
        "\n",
        "x = np.arange(len(categories))\n",
        "width = 0.35\n",
        "\n",
        "plt.style.use('dark_background')\n",
        "fig, ax = plt.subplots(figsize=(10, 6), dpi=300)\n",
        "fig.patch.set_facecolor('#0d1117')\n",
        "ax.set_facecolor('#0d1117')\n",
        "\n",
        "rects1 = ax.bar(x - width/2, baseline, width, label='Baseline (Single LightGBM)', color='#444c56')\n",
        "rects2 = ax.bar(x + width/2, ensemble, width, label='Ultimate Meta-Ensemble', color='#8a2be2')\n",
        "\n",
        "ax.set_ylabel('Financial Loss ($ USD)', fontsize=12, fontweight='bold', color='#c9d1d9')\n",
        "ax.set_title('Business Impact: Financial Loss Comparison', fontsize=16, fontweight='bold', color='#ffffff', pad=20)\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='#c9d1d9')\n",
        "ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=11, loc='upper left')\n",
        "\n",
        "def autolabel(rects):\n",
        "    for rect in rects:\n",
        "        height = rect.get_height()\n",
        "        ax.annotate(f'${height/1000:.0f}K',\n",
        "                    xy=(rect.get_x() + rect.get_width() / 2, height),\n",
        "                    xytext=(0, 5),\n",
        "                    textcoords='offset points',\n",
        "                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ffffff')\n",
        "\n",
        "autolabel(rects1)\n",
        "autolabel(rects2)\n",
        "\n",
        "# Highlight Savings\n",
        "ax.annotate('Net Savings:\\n$634,380', xy=(2.17, 3468566), xytext=(2.6, 3800000),\n",
        "            arrowprops=dict(facecolor='#3fb950', shrink=0.05, width=2, headwidth=8),\n",
        "            fontsize=13, fontweight='bold', color='#3fb950', ha='center', va='center',\n",
        "            bbox=dict(boxstyle='round,pad=0.5', fc='#161b22', ec='#3fb950', lw=2))\n",
        "\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "ax.spines['left'].set_color('#30363d')\n",
        "ax.spines['bottom'].set_color('#30363d')\n",
        "ax.tick_params(colors='#c9d1d9')\n",
        "ax.grid(axis='y', linestyle='--', alpha=0.2, color='#c9d1d9')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('08_Financial_Impact_ROI.png', facecolor=fig.get_facecolor(), edgecolor='none')\n",
        "plt.show()\n"
    ]

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_source
    }

    nb_data["cells"].extend([md_cell, code_cell])

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)
        
    print("Successfully appended new plot cell to visualizations/m5_total_visualizations.ipynb")

if __name__ == "__main__":
    generate_image()
    update_notebook()
