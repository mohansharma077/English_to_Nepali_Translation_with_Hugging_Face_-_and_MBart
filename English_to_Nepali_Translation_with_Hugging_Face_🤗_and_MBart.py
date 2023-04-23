{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yR0_MKOOeRNO",
        "outputId": "f063ba1d-ecab-4228-eb10-bb57af1473e1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.0/7.0 MB\u001b[0m \u001b[31m65.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.8/7.8 MB\u001b[0m \u001b[31m69.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m200.1/200.1 kB\u001b[0m \u001b[31m15.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ],
      "source": [
        "!pip install transformers -U -q"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2RtEvp3Me4va",
        "outputId": "134fb9c5-831c-4ae8-b779-dfcfe604890d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting sentencepiece\n",
            "  Downloading sentencepiece-0.1.98-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.3/1.3 MB\u001b[0m \u001b[31m53.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: sentencepiece\n",
            "Successfully installed sentencepiece-0.1.98\n"
          ]
        }
      ],
      "source": [
        "! pip install sentencepiece"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OLwUoKhxD-jt",
        "outputId": "39231415-f1aa-4ea8-b511-2c454878a378"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "transformers==4.28.1\n"
          ]
        }
      ],
      "source": [
        "!pip freeze | grep transformers"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "apok22zGeHWg"
      },
      "outputs": [],
      "source": [
        "from transformers import MBartForConditionalGeneration, MBart50TokenizerFast\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7aB7NFLzegiQ"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 113,
          "referenced_widgets": [
            "9e5985c7a6d14cbb876bb6e9ee344ffc",
            "d7d18d43b1a44d23afa6d00c7581798f",
            "562e7d8e75024e96a9f9a9e00fe077d5",
            "f178b8e5c27d4dda966bea1e2d510fb0",
            "0be05ebe6c6e4073b7b2e3634aa9e03b",
            "828d9e415a46442b9c9f4868092dbc1a",
            "6f13a69035394ae48d6064d41386f940",
            "850b62193444415b96af9c8daa322c12",
            "07c4faa510a540db91dfcc67a7ce8c66",
            "1b04eb5b22354a69b7552fa50e9742e6",
            "fef544cb6830451bb89878cff0bc175e",
            "991a80814b1a49f7a1bd41c6e52647d6",
            "f454a2fc918848a2a37d5be04d6253f2",
            "043e81ab2c0149498ad3d98633784624",
            "27240a21f48c43af910d2162733a7255",
            "591b2810912d4c52aaaa2136151e2165",
            "569df02611fb47f98802098eabc36e20",
            "a81f19936f1c49449cd57062d430927f",
            "fdce28bca5dd4774acde80efe1dca74c",
            "225c2cf75b3746f8aa5832d4915059c7",
            "24df8263ea16401480c97b1a147fb46b",
            "a247b1bba24f467f8c03367e41c85bbd",
            "6f903a3346914692af07a8a32f5c6f94",
            "5ed855c55b694a948c1a40480b0e0bea",
            "3f4fd704136b4ceb8d1dc5c0d8e52499",
            "da87a6ed3e204be7be91104500933ad7",
            "6a1cc2b6f4f04648bd72a043880bd0b2",
            "928c74ccb13d46b991b2b23db5eaf158",
            "8c0fc2c6dc2a432a8377ff760d0ccc6d",
            "f03d7bfefc54409aa0bd3f63d2468669",
            "b10240a36f554beda1645c6f3cd67354",
            "2bc48a83f570420fbeec63940820194c",
            "ed88fffdb4274985a19a3e73e7f774b7"
          ]
        },
        "id": "BQEzddKeeN2N",
        "outputId": "dbb2accf-430d-4305-f823-7104d7ede46b"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "9e5985c7a6d14cbb876bb6e9ee344ffc",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)lve/main/config.json:   0%|          | 0.00/1.43k [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "991a80814b1a49f7a1bd41c6e52647d6",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading pytorch_model.bin:   0%|          | 0.00/2.44G [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "6f903a3346914692af07a8a32f5c6f94",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)neration_config.json:   0%|          | 0.00/261 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "model = MBartForConditionalGeneration.from_pretrained(\"facebook/mbart-large-50-one-to-many-mmt\")\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 113,
          "referenced_widgets": [
            "585f558b78b84764810d1640b20c841e",
            "d8ddc0bb226b4567923ad348860f97cd",
            "1bb95c392f8843f0ace69033c557896b",
            "d84b092057eb493f95b246756b9e2fa1",
            "173c2c8e68ae4c72875ee9c0ffb2ec5c",
            "dcb4b7709cbe4e0ba791c75f4d963f40",
            "1c23b773498c47c4a73d973d1356daeb",
            "f01638e300b8498cb0731249f0173180",
            "702ac04377914d17a771cf3d818ce632",
            "450deb07493e49f8ab9dad0977066a46",
            "3ed5088cf38f4206a17e29441c424a2d",
            "14294cc4244f4a9d83dd4420b60640f1",
            "6019c47d55c9443f8ab3c70e2b3ebeba",
            "034bf7742399416db2e44bd57334afe6",
            "5f1d8c3d91fa491b88771a27e27f6591",
            "67e41333b49342438db2257dd1a481ad",
            "d6c285fefd1b4d24b17ffcb4cb321369",
            "f831dc62400d4acfa7fcd5c1fbad11f5",
            "b92dbc7d1aca4189b4c94d4aa7174789",
            "7fe23134ef3c4b39be2fc667abd0b272",
            "b5287db8a37741228fa5e4b8c4a1ec0f",
            "0eb5009674b0440fb4b13f7dcb1f7c3d",
            "9e03d817a833457db6d8dc49676c4dfc",
            "b4562fd6d331446caa20df34a9970cf7",
            "aaceaaa28d2e46a4b8962d95b71ad8f3",
            "3bb986adc7af44b0b4e873e0c1f7b1e2",
            "19c562002a834ad7bbe548989c78a5ef",
            "e514b277171547958263f531b9dfaf81",
            "e15832dd33b04cc9a91c17dc52e8b866",
            "d0cc6c549ac947ffa46f67f22dd0c98d",
            "45795e86f8354850ab8e863a5580e66b",
            "20747be79b4444d783661b41d9c0eb44",
            "8508da8fd03745328e1a4eb34473c84f"
          ]
        },
        "id": "NJPI14dwX2oC",
        "outputId": "0b22d8ea-6aa9-4864-d5d7-59155562a50f"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "585f558b78b84764810d1640b20c841e",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)okenizer_config.json:   0%|          | 0.00/528 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "14294cc4244f4a9d83dd4420b60640f1",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)tencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "9e03d817a833457db6d8dc49676c4dfc",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Downloading (…)cial_tokens_map.json:   0%|          | 0.00/717 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "tokenizer = MBart50TokenizerFast.from_pretrained(\"facebook/mbart-large-50-one-to-many-mmt\", src_lang=\"en_XX\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-mqJLyp4eZaj"
      },
      "outputs": [],
      "source": [
        "article_en = \"U.N encourages wearing masks\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Dzj93gO7f0Gc"
      },
      "outputs": [],
      "source": [
        "model_inputs = tokenizer(article_en, return_tensors=\"pt\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SAPso02IfehL",
        "outputId": "14025869-f2ac-4cdb-90a9-4372761bb6b2"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/transformers/generation/utils.py:1313: UserWarning: Using `max_length`'s default (200) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.\n",
            "  warnings.warn(\n"
          ]
        }
      ],
      "source": [
        "\n",
        "# translate from English to Nepali\n",
        "generated_tokens = model.generate(\n",
        "    **model_inputs,\n",
        "    forced_bos_token_id=tokenizer.lang_code_to_id[\"ne_NP\"]\n",
        ")\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kNRp5-dMfoiw"
      },
      "outputs": [],
      "source": [
        "translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0QUelm8ZgIOU",
        "outputId": "b9630b7a-f28d-4e7e-b3a9-f3591b0df11d"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "['संयुक्त राष्ट्रसंघले मास्क लगाउन प्रोत्साहित गर्दछ']"
            ]
          },
          "execution_count": 11,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "translation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9VGDlKnEgGKD",
        "outputId": "f6a4a58b-8def-4113-f92a-66395e26817e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m17.3/17.3 MB\u001b[0m \u001b[31m94.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m286.2/286.2 kB\u001b[0m \u001b[31m30.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.0/1.0 MB\u001b[0m \u001b[31m58.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m50.5/50.5 kB\u001b[0m \u001b[31m6.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m129.7/129.7 kB\u001b[0m \u001b[31m15.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m57.8/57.8 kB\u001b[0m \u001b[31m6.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m45.7/45.7 kB\u001b[0m \u001b[31m4.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m140.5/140.5 kB\u001b[0m \u001b[31m14.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m75.3/75.3 kB\u001b[0m \u001b[31m9.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m57.0/57.0 kB\u001b[0m \u001b[31m6.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m158.8/158.8 kB\u001b[0m \u001b[31m16.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m114.2/114.2 kB\u001b[0m \u001b[31m13.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m269.3/269.3 kB\u001b[0m \u001b[31m29.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m66.9/66.9 kB\u001b[0m \u001b[31m7.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m70.6/70.6 kB\u001b[0m \u001b[31m8.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m58.3/58.3 kB\u001b[0m \u001b[31m6.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Building wheel for ffmpy (setup.py) ... \u001b[?25l\u001b[?25hdone\n"
          ]
        }
      ],
      "source": [
        "!pip install -q gradio\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AB-hPS3IXsu1"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MLqLHHJYWqrL"
      },
      "outputs": [],
      "source": [
        "import gradio as gr"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "X35I2ivPWyAa"
      },
      "outputs": [],
      "source": [
        "def translate_gardio(input):\n",
        "  model_inputs = tokenizer(input, return_tensors=\"pt\")\n",
        "  generated_tokens = model.generate(\n",
        "    **model_inputs,\n",
        "    forced_bos_token_id=tokenizer.lang_code_to_id[\"ne_NP\"]\n",
        "    )\n",
        "  translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)\n",
        "  return translation[0]\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iQyTiL0XYQtW",
        "outputId": "16b2de9e-6dde-4fbf-9e81-c01792e5ca8c"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/gradio/inputs.py:27: UserWarning: Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.9/dist-packages/gradio/deprecation.py:40: UserWarning: `optional` parameter is deprecated, and it has no effect\n",
            "  warnings.warn(value)\n",
            "/usr/local/lib/python3.9/dist-packages/gradio/deprecation.py:40: UserWarning: `numeric` parameter is deprecated, and it has no effect\n",
            "  warnings.warn(value)\n",
            "/usr/local/lib/python3.9/dist-packages/gradio/deprecation.py:43: UserWarning: You have unused kwarg parameters in Interface, please remove them: {'buttons': button}\n",
            "  warnings.warn(\n"
          ]
        }
      ],
      "source": [
        "translate_interface=gr.Interface(fn=translate_gardio,\n",
        "                                 inputs=gr.inputs.Textbox(placeholder=\"Translate from English to Nepali\"),\n",
        "                                 buttons=gr.Button(value=\"Translate\"),\n",
        "                                 outputs=\"text\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/",
          "height": 648
        },
        "id": "ZFViYfO_ZWHN",
        "outputId": "e80bfcc4-e71b-4d5b-cf84-8b82b7ae1349"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().\n",
            "Running on public URL: https://ad70fef8690197c8b2.gradio.live\n",
            "\n",
            "This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "<div><iframe src=\"https://ad70fef8690197c8b2.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/transformers/generation/utils.py:1313: UserWarning: Using `max_length`'s default (200) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.\n",
            "  warnings.warn(\n"
          ]
        }
      ],
      "source": [
        "translate_interface.launch(debug=True,share=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7kbw0tlNZfWs"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "widgets": {
      "application/vnd.jupyter.widget-state+json": {
        "034bf7742399416db2e44bd57334afe6": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_b92dbc7d1aca4189b4c94d4aa7174789",
            "max": 5069051,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_7fe23134ef3c4b39be2fc667abd0b272",
            "value": 5069051
          }
        },
        "043e81ab2c0149498ad3d98633784624": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_fdce28bca5dd4774acde80efe1dca74c",
            "max": 2444714899,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_225c2cf75b3746f8aa5832d4915059c7",
            "value": 2444714899
          }
        },
        "07c4faa510a540db91dfcc67a7ce8c66": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "0be05ebe6c6e4073b7b2e3634aa9e03b": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "0eb5009674b0440fb4b13f7dcb1f7c3d": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "14294cc4244f4a9d83dd4420b60640f1": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_6019c47d55c9443f8ab3c70e2b3ebeba",
              "IPY_MODEL_034bf7742399416db2e44bd57334afe6",
              "IPY_MODEL_5f1d8c3d91fa491b88771a27e27f6591"
            ],
            "layout": "IPY_MODEL_67e41333b49342438db2257dd1a481ad"
          }
        },
        "173c2c8e68ae4c72875ee9c0ffb2ec5c": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "19c562002a834ad7bbe548989c78a5ef": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "1b04eb5b22354a69b7552fa50e9742e6": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "1bb95c392f8843f0ace69033c557896b": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_f01638e300b8498cb0731249f0173180",
            "max": 528,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_702ac04377914d17a771cf3d818ce632",
            "value": 528
          }
        },
        "1c23b773498c47c4a73d973d1356daeb": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "20747be79b4444d783661b41d9c0eb44": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "225c2cf75b3746f8aa5832d4915059c7": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "24df8263ea16401480c97b1a147fb46b": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "27240a21f48c43af910d2162733a7255": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_24df8263ea16401480c97b1a147fb46b",
            "placeholder": "​",
            "style": "IPY_MODEL_a247b1bba24f467f8c03367e41c85bbd",
            "value": " 2.44G/2.44G [00:09&lt;00:00, 245MB/s]"
          }
        },
        "2bc48a83f570420fbeec63940820194c": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "3bb986adc7af44b0b4e873e0c1f7b1e2": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_20747be79b4444d783661b41d9c0eb44",
            "placeholder": "​",
            "style": "IPY_MODEL_8508da8fd03745328e1a4eb34473c84f",
            "value": " 717/717 [00:00&lt;00:00, 45.8kB/s]"
          }
        },
        "3ed5088cf38f4206a17e29441c424a2d": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "3f4fd704136b4ceb8d1dc5c0d8e52499": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_f03d7bfefc54409aa0bd3f63d2468669",
            "max": 261,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_b10240a36f554beda1645c6f3cd67354",
            "value": 261
          }
        },
        "450deb07493e49f8ab9dad0977066a46": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "45795e86f8354850ab8e863a5580e66b": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "562e7d8e75024e96a9f9a9e00fe077d5": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_850b62193444415b96af9c8daa322c12",
            "max": 1428,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_07c4faa510a540db91dfcc67a7ce8c66",
            "value": 1428
          }
        },
        "569df02611fb47f98802098eabc36e20": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "585f558b78b84764810d1640b20c841e": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_d8ddc0bb226b4567923ad348860f97cd",
              "IPY_MODEL_1bb95c392f8843f0ace69033c557896b",
              "IPY_MODEL_d84b092057eb493f95b246756b9e2fa1"
            ],
            "layout": "IPY_MODEL_173c2c8e68ae4c72875ee9c0ffb2ec5c"
          }
        },
        "591b2810912d4c52aaaa2136151e2165": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "5ed855c55b694a948c1a40480b0e0bea": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_928c74ccb13d46b991b2b23db5eaf158",
            "placeholder": "​",
            "style": "IPY_MODEL_8c0fc2c6dc2a432a8377ff760d0ccc6d",
            "value": "Downloading (…)neration_config.json: 100%"
          }
        },
        "5f1d8c3d91fa491b88771a27e27f6591": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_b5287db8a37741228fa5e4b8c4a1ec0f",
            "placeholder": "​",
            "style": "IPY_MODEL_0eb5009674b0440fb4b13f7dcb1f7c3d",
            "value": " 5.07M/5.07M [00:00&lt;00:00, 44.2MB/s]"
          }
        },
        "6019c47d55c9443f8ab3c70e2b3ebeba": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_d6c285fefd1b4d24b17ffcb4cb321369",
            "placeholder": "​",
            "style": "IPY_MODEL_f831dc62400d4acfa7fcd5c1fbad11f5",
            "value": "Downloading (…)tencepiece.bpe.model: 100%"
          }
        },
        "67e41333b49342438db2257dd1a481ad": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "6a1cc2b6f4f04648bd72a043880bd0b2": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "6f13a69035394ae48d6064d41386f940": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "6f903a3346914692af07a8a32f5c6f94": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_5ed855c55b694a948c1a40480b0e0bea",
              "IPY_MODEL_3f4fd704136b4ceb8d1dc5c0d8e52499",
              "IPY_MODEL_da87a6ed3e204be7be91104500933ad7"
            ],
            "layout": "IPY_MODEL_6a1cc2b6f4f04648bd72a043880bd0b2"
          }
        },
        "702ac04377914d17a771cf3d818ce632": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "7fe23134ef3c4b39be2fc667abd0b272": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "828d9e415a46442b9c9f4868092dbc1a": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "8508da8fd03745328e1a4eb34473c84f": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "850b62193444415b96af9c8daa322c12": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "8c0fc2c6dc2a432a8377ff760d0ccc6d": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "928c74ccb13d46b991b2b23db5eaf158": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "991a80814b1a49f7a1bd41c6e52647d6": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_f454a2fc918848a2a37d5be04d6253f2",
              "IPY_MODEL_043e81ab2c0149498ad3d98633784624",
              "IPY_MODEL_27240a21f48c43af910d2162733a7255"
            ],
            "layout": "IPY_MODEL_591b2810912d4c52aaaa2136151e2165"
          }
        },
        "9e03d817a833457db6d8dc49676c4dfc": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_b4562fd6d331446caa20df34a9970cf7",
              "IPY_MODEL_aaceaaa28d2e46a4b8962d95b71ad8f3",
              "IPY_MODEL_3bb986adc7af44b0b4e873e0c1f7b1e2"
            ],
            "layout": "IPY_MODEL_19c562002a834ad7bbe548989c78a5ef"
          }
        },
        "9e5985c7a6d14cbb876bb6e9ee344ffc": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_d7d18d43b1a44d23afa6d00c7581798f",
              "IPY_MODEL_562e7d8e75024e96a9f9a9e00fe077d5",
              "IPY_MODEL_f178b8e5c27d4dda966bea1e2d510fb0"
            ],
            "layout": "IPY_MODEL_0be05ebe6c6e4073b7b2e3634aa9e03b"
          }
        },
        "a247b1bba24f467f8c03367e41c85bbd": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "a81f19936f1c49449cd57062d430927f": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "aaceaaa28d2e46a4b8962d95b71ad8f3": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_d0cc6c549ac947ffa46f67f22dd0c98d",
            "max": 717,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_45795e86f8354850ab8e863a5580e66b",
            "value": 717
          }
        },
        "b10240a36f554beda1645c6f3cd67354": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "b4562fd6d331446caa20df34a9970cf7": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_e514b277171547958263f531b9dfaf81",
            "placeholder": "​",
            "style": "IPY_MODEL_e15832dd33b04cc9a91c17dc52e8b866",
            "value": "Downloading (…)cial_tokens_map.json: 100%"
          }
        },
        "b5287db8a37741228fa5e4b8c4a1ec0f": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "b92dbc7d1aca4189b4c94d4aa7174789": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "d0cc6c549ac947ffa46f67f22dd0c98d": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "d6c285fefd1b4d24b17ffcb4cb321369": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "d7d18d43b1a44d23afa6d00c7581798f": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_828d9e415a46442b9c9f4868092dbc1a",
            "placeholder": "​",
            "style": "IPY_MODEL_6f13a69035394ae48d6064d41386f940",
            "value": "Downloading (…)lve/main/config.json: 100%"
          }
        },
        "d84b092057eb493f95b246756b9e2fa1": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_450deb07493e49f8ab9dad0977066a46",
            "placeholder": "​",
            "style": "IPY_MODEL_3ed5088cf38f4206a17e29441c424a2d",
            "value": " 528/528 [00:00&lt;00:00, 19.2kB/s]"
          }
        },
        "d8ddc0bb226b4567923ad348860f97cd": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_dcb4b7709cbe4e0ba791c75f4d963f40",
            "placeholder": "​",
            "style": "IPY_MODEL_1c23b773498c47c4a73d973d1356daeb",
            "value": "Downloading (…)okenizer_config.json: 100%"
          }
        },
        "da87a6ed3e204be7be91104500933ad7": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_2bc48a83f570420fbeec63940820194c",
            "placeholder": "​",
            "style": "IPY_MODEL_ed88fffdb4274985a19a3e73e7f774b7",
            "value": " 261/261 [00:00&lt;00:00, 8.74kB/s]"
          }
        },
        "dcb4b7709cbe4e0ba791c75f4d963f40": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "e15832dd33b04cc9a91c17dc52e8b866": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "e514b277171547958263f531b9dfaf81": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "ed88fffdb4274985a19a3e73e7f774b7": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "f01638e300b8498cb0731249f0173180": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "f03d7bfefc54409aa0bd3f63d2468669": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "f178b8e5c27d4dda966bea1e2d510fb0": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_1b04eb5b22354a69b7552fa50e9742e6",
            "placeholder": "​",
            "style": "IPY_MODEL_fef544cb6830451bb89878cff0bc175e",
            "value": " 1.43k/1.43k [00:00&lt;00:00, 62.9kB/s]"
          }
        },
        "f454a2fc918848a2a37d5be04d6253f2": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_569df02611fb47f98802098eabc36e20",
            "placeholder": "​",
            "style": "IPY_MODEL_a81f19936f1c49449cd57062d430927f",
            "value": "Downloading pytorch_model.bin: 100%"
          }
        },
        "f831dc62400d4acfa7fcd5c1fbad11f5": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "fdce28bca5dd4774acde80efe1dca74c": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "fef544cb6830451bb89878cff0bc175e": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        }
      }
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}