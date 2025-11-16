import gradio as gr
import os
import argparse
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Suppress verbose httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)
from datetime import datetime
import torchaudio
import librosa
import soundfile as sf

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types
from whisper_wrapper import WhisperWrapper

# Configure logging
logger = logging.getLogger(__name__)

# Save audio to temporary directory
def save_audio(audio_type, audio_data, sr, tmp_dir):
    """Save audio data to a temporary file with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(tmp_dir, audio_type, f"{current_time}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(save_path, audio_data, sr)
        logger.debug(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


class EditxTab:
    """Audio editing and voice cloning interface tab"""

    def __init__(self, args):
        self.args = args
        self.edit_type_list = list(get_supported_edit_types().keys())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enable_auto_transcribe = getattr(args, 'enable_auto_transcribe', False)

    def history_messages_to_show(self, messages):
        """Convert message history to gradio chatbot format"""
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": gr.Audio(value=raw_audio_part, interactive=False)},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": gr.Audio(value=edit_audio_part, interactive=False)}
            ])
        return show_msgs

    def generate_clone(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate cloned audio"""
        self.logger.info("Starting voice cloning process")
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Use common_tts_engine for cloning
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed"
                self.logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        
    def generate_edit(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate edited audio"""
        self.logger.info("Starting audio editing process")

        # Input validation
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                self.logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                sample_rate, audio_numpy, previous_text = state["history_audio"][-1]
                temp_path = save_audio("temp", audio_numpy, sample_rate, self.args.tmp_dir)
                audio_to_edit = temp_path
                text_to_use = previous_text
                self.logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # For para-linguistic, use generated_text; otherwise use source text
            if edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, edit_type, edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                self.logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }

    def register_components(self):
        """Register gradio components - maintaining exact layout from original"""
        with gr.Tab("Editx"):
            with gr.Row():
                with gr.Column():
                    self.model_input = gr.Textbox(label="Model Name", value="Step-Audio-EditX", scale=1)
                    self.prompt_text_input = gr.Textbox(label="Prompt Text", value="", scale=1)
                    self.prompt_audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        format="wav",
                        type="filepath",
                        label="Input Audio",
                    )
                    self.generated_text = gr.Textbox(label="Target Text", lines=1, max_lines=200, max_length=1000)
                with gr.Column():
                    with gr.Row():
                        self.edit_type = gr.Dropdown(label="Task", choices=self.edit_type_list, value="clone")
                        self.edit_info = gr.Dropdown(label="Sub-task", choices=[], value=None)
                    self.chat_box = gr.Chatbot(label="History", type="messages", height=480*1)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.button_tts = gr.Button("CLONE", variant="primary")
                        self.button_edit = gr.Button("EDIT", variant="primary")
                with gr.Column():
                    self.clean_history_submit = gr.Button("Clear History", variant="primary")

            gr.Markdown("---")
            gr.Markdown("""
                **Button Description:**
                - CLONE: Synthesizes audio based on uploaded audio and text, only used for clone mode, will clear history information when used.
                - EDIT: Edits based on uploaded audio, or continues to stack edit effects based on the previous round of generated audio.
                """)
            gr.Markdown("""
                **Operation Workflow:**
                - Upload the audio to be edited on the left side. The text will be **automatically transcribed** using Whisper if the Prompt Text field is empty;
                - If the task requires modifying text content (such as clone, para-linguistic), fill in the text to be synthesized in the "Target Text" field. For all other tasks, keep the uploaded audio text content unchanged;
                - Select tasks and subtasks on the right side (some tasks have no subtasks, such as vad, etc.);
                - Click the "CLONE" or "EDIT" button on the left side, and audio will be generated in the dialog box on the right side.
                
                **💡 Tip:** Leave "Prompt Text" empty when uploading audio to auto-transcribe it!
                """)
            gr.Markdown("""
                **Para-linguistic Description:**
                - Supported tags include: [Breathing] [Laughter] [Surprise-oh] [Confirmation-en] [Uhm] [Surprise-ah] [Surprise-wa] [Sigh] [Question-ei] [Dissatisfaction-hnn]
                - Example:
                    - Fill in "clone text" field: "Great, the weather is so nice today." Click the "CLONE" button to get audio.
                    - Change "clone text" field to: "Great[Laughter], the weather is so nice today[Surprise-ah]." Click the "EDIT" button to get para-linguistic audio.
                """)

    def register_events(self):
        """Register event handlers"""
        # Create independent state for each session
        state = gr.State(self.init_state())

        self.button_tts.click(self.generate_clone,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])
        self.button_edit.click(self.generate_edit,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])

        self.clean_history_submit.click(self.clear_history, inputs=[state], outputs=[self.chat_box, state])
        self.edit_type.change(
            fn=self.update_edit_info,
            inputs=self.edit_type,
            outputs=self.edit_info,
        )

        # Add audio transcription event only if enabled
        if self.enable_auto_transcribe:
            self.prompt_audio_input.change(
                fn=self.transcribe_audio,
                inputs=[self.prompt_audio_input, self.prompt_text_input],
                outputs=self.prompt_text_input,
            )

    def update_edit_info(self, category):
        """Update sub-task dropdown based on main task selection"""
        category_items = get_supported_edit_types()
        choices = category_items.get(category, [])
        value = None if len(choices) == 0 else choices[0]
        return gr.Dropdown(label="Sub-task", choices=choices, value=value)

    def transcribe_audio(self, audio_input, current_text):
        """Transcribe audio using Whisper ASR when prompt text is empty"""
        # Only transcribe if current text is empty
        if current_text and current_text.strip():
            return current_text  # Keep existing text
        if not audio_input:
            return ""  # No audio to transcribe
        if whisper_asr is None:
            self.logger.error("Whisper ASR not initialized.")
            return ""

        try:
            # Transcribe audio
            transcribed_text = whisper_asr(audio_input)
            self.logger.info(f"Audio transcribed: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            return ""


def launch_demo(args, editx_tab):
    """Launch the gradio demo"""
    with gr.Blocks(
            theme=gr.themes.Soft(), 
            title="🎙️ Step-Audio-EditX",
            css="""
    :root {
        --font: "Helvetica Neue", Helvetica, Arial, sans-serif;
        --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    """) as demo:
        gr.Markdown("## 🎙️ Step-Audio-EditX")
        gr.Markdown("Audio Editing and Zero-Shot Cloning using Step-Audio-EditX")

        # Register components
        editx_tab.register_components()

        # Register events
        editx_tab.register_events()

    # Launch demo
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share if hasattr(args, 'share') else False
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Step-Audio Edit Demo")
    parser.add_argument("--model-path", type=str, default="./models/Step-Audio-EditX", help="Model path (default: ./models/Step-Audio-EditX)")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Demo server name.")
    parser.add_argument("--server-port", type=int, default=7860, help="Demo server port.")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/gradio", help="Save path.")
    
    # Quantization mode
    parser.add_argument(
        "--quantize-model",
        action="store_true",
        help="Run quantization on the model instead of launching the demo"
    )
    parser.add_argument(
        "--quantize-output-suffix",
        type=str,
        default="awq-4bit",
        help="Output subdirectory name for quantized model (default: awq-4bit)"
    )
    parser.add_argument(
        "--quantize-dataset",
        type=str,
        default="open_platypus",
        help="Calibration dataset for quantization (default: open_platypus)"
    )
    parser.add_argument(
        "--quantize-scheme",
        type=str,
        default="",
        choices=["", "W4A16_ASYM", "W4A16_SYM", "W8A16"],
        help="Quantization scheme (default: W4A16_ASYM)"
    )
    parser.add_argument(
        "--quantize-group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)"
    )
    parser.add_argument(
        "--quantize-max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length for quantization (default: 4096)"
    )
    parser.add_argument(
        "--quantize-num-samples",
        type=int,
        default=512,
        help="Number of calibration samples for quantization (default: 512)"
    )
    
    # Model download mode
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download model from HuggingFace or ModelScope instead of launching the demo"
    )
    parser.add_argument(
        "--download-hub",
        type=str,
        default="hf",
        choices=["hf", "ms"],
        help="Model hub to download from: hf (HuggingFace) or ms (ModelScope) (default: hf)"
    )
    parser.add_argument(
        "--download-repo-id",
        type=str,
        default=None,
        help="Repository ID to download (e.g., 'stepfun-ai/Step-Audio-EditX')"
    )
    parser.add_argument(
        "--download-output-dir",
        type=str,
        default=None,
        help="Output directory for downloaded model (default: ./models/<repo_name>)"
    )
    parser.add_argument(
        "--download-revision",
        type=str,
        default=None,
        help="Model revision/branch to download (default: main)"
    )
    
    # Auto-setup mode
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download model if it doesn't exist (default: disabled)"
    )
    parser.add_argument(
        "--auto-quantize",
        action="store_true",
        help="Automatically quantize model after download if quantized version doesn't exist (default: disabled)"
    )
    parser.add_argument(
        "--default-repo-id",
        type=str,
        default="stepfun-ai/Step-Audio-EditX",
        help="Default repository ID for auto-download (default: stepfun-ai/Step-Audio-EditX)"
    )
    
    parser.add_argument("--share", action="store_true", help="Share gradio app.")

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="Tokenizer model ID for online loading"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID for online loading (if different from model-path)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int4", "int8", "awq-4bit", "bnb-4bit"],
        help="Enable quantization for the TTS model to reduce memory usage."
             "Choices: int4 (BnB 4-bit), int8 (BnB 8-bit), awq-4bit (AWQ 4-bit), bnb-4bit (use pre-quantized BnB model)."
             "When quantization is enabled, data types are handled automatically by the quantization library."
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch data type for model operations. This setting only applies when quantization is disabled. "
             "When quantization is enabled, data types are managed automatically."
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "balanced", "balanced_low_0"],
        help="Device mapping for model loading. 'auto' enables CPU offloading for lower memory usage (default: auto)"
    )
    parser.add_argument(
        "--enable-auto-transcribe",
        action="store_true",
        default=True,
        help="Enable automatic audio transcription when uploading audio files (default: enabled)"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-tiny",
        choices=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"],
        help="Whisper model to use for auto-transcription (default: tiny for speed)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable CPU offloading to reduce GPU memory usage (uses ~8GB VRAM by default, configurable with --max-gpu-memory)"
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        default="6GiB",
        help="Maximum GPU memory to use in low-memory mode (default: 6GiB). Values above 6GiB are automatically capped to prevent OOM during inference. 6GiB is optimal, using ~13GB total VRAM."
    )

    args = parser.parse_args()

    # If download mode is enabled, download model and exit
    if args.download_model:
        logger.info("=" * 60)
        logger.info("MODEL DOWNLOAD MODE")
        logger.info("=" * 60)
        
        if not args.download_repo_id:
            logger.error("Error: --download-repo-id is required when using --download-model")
            logger.info("Example: python app.py --download-model --download-repo-id stepfun-ai/Step-Audio-EditX")
            exit(1)
        
        try:
            # Determine output directory
            if args.download_output_dir:
                output_dir = args.download_output_dir
            else:
                # Extract repo name from repo_id
                repo_name = args.download_repo_id.split("/")[-1]
                output_dir = os.path.join("./models", repo_name)
            
            logger.info(f"Repository ID: {args.download_repo_id}")
            logger.info(f"Model hub: {args.download_hub}")
            logger.info(f"Output directory: {output_dir}")
            if args.download_revision:
                logger.info(f"Revision: {args.download_revision}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Download based on hub
            if args.download_hub == "hf":
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    logger.error("huggingface_hub is required for downloading from HuggingFace.")
                    logger.error("Install it with: pip install huggingface_hub")
                    exit(1)
                
                logger.info("Downloading from HuggingFace...")
                downloaded_path = snapshot_download(
                    repo_id=args.download_repo_id,
                    revision=args.download_revision,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
                
            elif args.download_hub == "ms":
                try:
                    from modelscope.hub.snapshot_download import snapshot_download
                except ImportError:
                    logger.error("modelscope is required for downloading from ModelScope.")
                    logger.error("Install it with: pip install modelscope")
                    exit(1)
                
                logger.info("Downloading from ModelScope...")
                downloaded_path = snapshot_download(
                    args.download_repo_id,
                    revision=args.download_revision,
                    cache_dir=output_dir
                )
            
            logger.info("=" * 60)
            logger.info("MODEL DOWNLOAD COMPLETED SUCCESSFULLY!")
            logger.info(f"Model saved to: {output_dir}")
            logger.info("=" * 60)
            logger.info(f"\nTo use this model, run:")
            logger.info(f"python app.py --model-path {output_dir}")
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            exit(1)
        
        # Exit after download
        exit(0)

    # If quantization mode is enabled, run quantization and exit
    if args.quantize_model:
        from quantization.awq_quantize import quantize_model, validate_model_path, create_output_directory
        
        logger.info("=" * 60)
        logger.info("QUANTIZATION MODE")
        logger.info("=" * 60)
        
        try:
            # Validate model path
            model_path = validate_model_path(args.model_path)
            
            # Create output directory
            output_dir = create_output_directory(model_path, args.quantize_output_suffix)
            
            logger.info(f"Model path: {model_path}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Quantization scheme: {args.quantize_scheme}")
            logger.info(f"Group size: {args.quantize_group_size}")
            logger.info(f"Calibration dataset: {args.quantize_dataset}")
            logger.info(f"Calibration samples: {args.quantize_num_samples}")
            logger.info(f"Max sequence length: {args.quantize_max_seq_length}")
            
            # Perform quantization
            quantize_model(
                model_path=str(model_path),
                output_dir=str(output_dir),
                scheme=args.quantize_scheme,
                dataset=args.quantize_dataset,
                max_seq_length=args.quantize_max_seq_length,
                num_calibration_samples=args.quantize_num_samples,
                group_size=args.quantize_group_size,
                ignore_layers=["lm_head", "embed_tokens", "model.embed_tokens", "model.norm", "norm", "output", "classifier"],
                device=None  # Auto-detect
            )
            
            logger.info("=" * 60)
            logger.info("QUANTIZATION COMPLETED SUCCESSFULLY!")
            logger.info(f"Quantized model saved to: {output_dir}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            exit(1)
        
        # Exit after quantization
        exit(0)

    # Auto-download and auto-quantize logic
    if args.auto_download or args.auto_quantize:
        # Check if both required model directories exist
        tokenizer_path = os.path.join(args.model_path, "Step-Audio-Tokenizer")
        editx_path = os.path.join(args.model_path, "Step-Audio-EditX")
        model_exists = os.path.exists(tokenizer_path) and os.path.exists(editx_path)
        
        if model_exists:
            logger.info("✓ Models already downloaded, skipping download")
        
        # Check if we need to download
        if args.auto_download and not model_exists:
            logger.info("=" * 60)
            logger.info("AUTO-DOWNLOAD: Models not found, downloading...")
            logger.info("=" * 60)
            
            try:
                from huggingface_hub import snapshot_download
                
                os.makedirs(args.model_path, exist_ok=True)
                
                # Download Step-Audio-Tokenizer
                if not os.path.exists(tokenizer_path):
                    logger.info("Downloading Step-Audio-Tokenizer...")
                    snapshot_download(
                        repo_id="stepfun-ai/Step-Audio-Tokenizer",
                        local_dir=tokenizer_path,
                        local_dir_use_symlinks=False
                    )
                    logger.info("✓ Step-Audio-Tokenizer downloaded!")
                
                # Download Step-Audio-EditX
                if not os.path.exists(editx_path):
                    logger.info("Downloading Step-Audio-EditX (this may take a while, ~7GB)...")
                    snapshot_download(
                        repo_id=args.default_repo_id,
                        local_dir=editx_path,
                        local_dir_use_symlinks=False
                    )
                    logger.info("✓ Step-Audio-EditX downloaded!")
                
                logger.info("✓ All models downloaded successfully!")
                model_exists = True
                
            except ImportError:
                logger.error("huggingface_hub is required for auto-download.")
                logger.error("Install it with: pip install huggingface_hub")
                exit(1)
            except Exception as e:
                logger.error(f"Auto-download failed: {e}")
                exit(1)
        
        # Check if we need to quantize (only quantize the EditX model, not tokenizer)
        if args.auto_quantize and model_exists:
            editx_model_path = os.path.join(args.model_path, "Step-Audio-EditX")
            
            # Check for any existing quantized model (awq-4bit only)
            # Note: BnB quantization is applied on-the-fly, not pre-saved
            awq_path = os.path.join(editx_model_path, "awq-4bit")
            quantized_path = os.path.join(editx_model_path, args.quantize_output_suffix)
            
            if os.path.exists(awq_path):
                logger.info("✓ AWQ-4bit quantized model found, will use it")
                if not args.quantization:
                    args.quantization = "awq-4bit"
            elif os.path.exists(quantized_path):
                logger.info("✓ Quantized model already exists, skipping quantization")
            
            if not os.path.exists(quantized_path) and not os.path.exists(bnb_path) and not os.path.exists(awq_path):
                logger.info("=" * 60)
                logger.info("AUTO-QUANTIZE: Quantized model not found, quantizing...")
                logger.info("=" * 60)
                
                try:
                    from quantization.awq_quantize import quantize_model, validate_model_path, create_output_directory
                    
                    model_path = validate_model_path(editx_model_path)
                    output_dir = create_output_directory(model_path, args.quantize_output_suffix)
                    
                    logger.info(f"Quantizing model at: {model_path}")
                    logger.info(f"Output directory: {output_dir}")
                    
                    quantize_model(
                        model_path=str(model_path),
                        output_dir=str(output_dir),
                        scheme=args.quantize_scheme,
                        dataset=args.quantize_dataset,
                        max_seq_length=args.quantize_max_seq_length,
                        num_calibration_samples=args.quantize_num_samples,
                        group_size=args.quantize_group_size,
                        ignore_layers=["lm_head", "embed_tokens", "model.embed_tokens", "model.norm", "norm", "output", "classifier"],
                        device=None
                    )
                    
                    logger.info("✓ Model quantized successfully!")
                    # Set quantization flag so the model loader knows to use quantized version
                    if not args.quantization:
                        args.quantization = "awq-4bit"
                    
                except Exception as e:
                    logger.error(f"Auto-quantize failed: {e}")
                    logger.info("Continuing with non-quantized model...")
            else:
                # Quantized model already exists, set flag to use it
                if not args.quantization:
                    args.quantization = "awq-4bit"

    # Map string arguments to actual types
    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    
    # If model exists locally, use LOCAL source to avoid path issues
    if os.path.exists(args.model_path) and os.path.isdir(args.model_path):
        model_source = ModelSource.LOCAL
        logger.info("Model found locally, using LOCAL source")
    else:
        model_source = source_mapping[args.model_source]

    # Map torch dtype string to actual torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping[args.torch_dtype]

    logger.info(f"Loading models with source: {args.model_source}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer model ID: {args.tokenizer_model_id}")
    logger.info(f"Torch dtype: {args.torch_dtype}")
    logger.info(f"Device map: {args.device_map}")
    if args.tts_model_id:
        logger.info(f"TTS model ID: {args.tts_model_id}")
    if args.quantization:
        logger.info(f"🔧 {args.quantization.upper()} quantization enabled")

    # Initialize models
    try:
        # Determine the correct paths based on directory structure
        nested_tokenizer = os.path.join(args.model_path, "Step-Audio-Tokenizer")
        nested_tts = os.path.join(args.model_path, "Step-Audio-EditX")
        
        # Check for nested structure first (recommended)
        if os.path.exists(nested_tokenizer) and os.path.exists(nested_tts):
            logger.info("Using nested directory structure")
            tokenizer_path = nested_tokenizer
            
            # Handle different quantization types
            if args.quantization == "awq-4bit":
                quantized_tts = os.path.join(nested_tts, "awq-4bit")
                if os.path.exists(quantized_tts):
                    tts_path = nested_tts  # Use base path, model_loader will add awq-4bit
                    logger.info("Using AWQ quantized TTS model")
                else:
                    tts_path = nested_tts
                    logger.warning("AWQ quantization requested but quantized model not found, using original")
                    args.quantization = None  # Disable quantization flag
            elif args.quantization == "bnb-4bit":
                # Always use the original model and apply BnB quantization on load
                # Pre-saved quantized models often get dequantized when loaded
                tts_path = nested_tts
                args.quantization = "int4"  # Apply BnB 4-bit quantization on load
                logger.info("Using BitsAndBytes 4-bit quantization (applied on load)")
            else:
                tts_path = nested_tts
        else:
            # Check if models are in parent directory (sibling structure)
            parent_dir = os.path.dirname(args.model_path)
            sibling_tokenizer = os.path.join(parent_dir, "Step-Audio-Tokenizer")
            sibling_tts = os.path.join(parent_dir, "Step-Audio-EditX")
            
            if os.path.exists(sibling_tokenizer) and os.path.exists(sibling_tts):
                logger.info("Using sibling directory structure")
                tokenizer_path = sibling_tokenizer
                
                # Handle different quantization types
                if args.quantization == "awq-4bit":
                    quantized_tts = os.path.join(sibling_tts, "awq-4bit")
                    if os.path.exists(quantized_tts):
                        tts_path = sibling_tts
                        logger.info("Using AWQ quantized TTS model")
                    else:
                        tts_path = sibling_tts
                        logger.warning("AWQ quantization requested but quantized model not found, using original")
                        args.quantization = None
                elif args.quantization == "bnb-4bit":
                    tts_path = sibling_tts
                    args.quantization = "int4"
                    logger.info("Using BitsAndBytes 4-bit quantization (applied on load)")
                else:
                    tts_path = sibling_tts
            else:
                # No valid structure found
                logger.warning("Expected directory structure not found")
                logger.warning(f"Expected nested: {args.model_path}/Step-Audio-Tokenizer and {args.model_path}/Step-Audio-EditX")
                logger.warning(f"Or sibling: {parent_dir}/Step-Audio-Tokenizer and {parent_dir}/Step-Audio-EditX")
                logger.error("Please run with --auto-download to download the correct model structure")
                exit(1)
        
        # Load StepAudioTokenizer
        encoder = StepAudioTokenizer(
            tokenizer_path,
            model_source=model_source,
            funasr_model_id=args.tokenizer_model_id
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")

        # Configure memory limits for low-memory mode
        if args.low_memory:
            # Parse the requested memory value
            import re
            mem_match = re.match(r'(\d+)([GMK]i?B)', args.max_gpu_memory)
            if mem_match:
                mem_value = int(mem_match.group(1))
                mem_unit = mem_match.group(2)
                
                # Cap at 6GiB for stability - higher values cause OOM during inference
                # This is because more GPU memory = more layers on GPU = larger activations
                if mem_unit in ['GiB', 'GB'] and mem_value > 6:
                    actual_memory = f"6{mem_unit}"
                    logger.warning(f"⚠️  Requested {args.max_gpu_memory} but capping at {actual_memory}")
                    logger.warning(f"⚠️  Higher values load more layers on GPU, causing OOM during inference")
                    logger.warning(f"⚠️  6GiB is optimal - uses ~13GB total VRAM with CPU offloading")
                else:
                    actual_memory = args.max_gpu_memory
            else:
                actual_memory = args.max_gpu_memory
            
            max_memory = {0: actual_memory, "cpu": "32GiB"}
            device_map_to_use = "auto"
            logger.info(f"🔧 Low-memory mode: GPU limited to {actual_memory}, CPU/RAM to 32GB")
            logger.info(f"🔧 max_memory: {max_memory}")
        else:
            max_memory = None
            device_map_to_use = args.device_map

        # Initialize common TTS engine directly
        common_tts_engine = StepAudioTTS(
            tts_path,
            encoder,
            model_source=model_source,
            tts_model_id=args.tts_model_id,
            quantization_config=args.quantization,
            torch_dtype=torch_dtype,
            device_map=device_map_to_use,
            max_memory=max_memory
        )
        logger.info("✓ StepCommonAudioTTS loaded successfully")
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        
        if args.enable_auto_transcribe:
            whisper_asr = WhisperWrapper(model_id=args.whisper_model)
            logger.info(f"✓ Automatic audio transcription enabled (using {args.whisper_model})")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        exit(1)

    # Create EditxTab instance
    editx_tab = EditxTab(args)

    # Launch demo
    launch_demo(args, editx_tab)
