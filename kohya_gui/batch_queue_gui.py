import os
import sys
import gradio as gr
from typing import List
from .common_gui import (
    list_files,
    create_refresh_button,
    get_file_path,
    setup_environment,
    scriptdir,
)
from .class_command_executor import CommandExecutor
from .custom_logging import setup_logging

log = setup_logging()
PYTHON = sys.executable
TRAIN_SCRIPT = os.path.join(scriptdir, "sd-scripts", "train_network.py")


def list_config_files(path: str) -> List[str]:
    return list(list_files(path, exts=[".toml"], all=True))


class TrainingQueue:
    def __init__(self, headless: bool = False):
        self.queue: List[str] = []
        self.executor = CommandExecutor(headless=headless)

    def add(self, config_file: str) -> str:
        if config_file and os.path.exists(config_file):
            self.queue.append(config_file)
            log.info(f"Added {config_file} to queue")
        else:
            log.info(f"Config file not found: {config_file}")
        return "\n".join(self.queue)

    def clear(self) -> str:
        self.queue.clear()
        log.info("Queue cleared")
        return ""

    def run_queue(self) -> str:
        env = setup_environment()
        while self.queue:
            config = self.queue[0]
            run_cmd = [PYTHON, TRAIN_SCRIPT, "--config_file", config]
            self.executor.execute_command(run_cmd=run_cmd, env=env)
            self.executor.wait_for_training_to_end()
            self.queue.pop(0)
        log.info("Queue finished")
        return ""


def gradio_batch_queue_tab(headless: bool = False):
    current_config_dir = os.path.join(scriptdir, "presets")

    def list_dirs(path):
        nonlocal current_config_dir
        current_config_dir = path
        return list_config_files(path)

    trainer = TrainingQueue(headless=headless)

    with gr.Tab("Batch Queue"):
        gr.Markdown("Add multiple training config files to a queue and run them sequentially.")
        with gr.Group(), gr.Row():
            config_file = gr.Dropdown(
                label="Config file",
                choices=[""] + list_dirs(current_config_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(
                config_file,
                lambda: None,
                lambda: {"choices": [""] + list_dirs(current_config_dir)},
                "open_folder_small",
            )
            open_button = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            open_button.click(
                get_file_path,
                outputs=config_file,
                show_progress=False,
            )

        queue_box = gr.Textbox(label="Queue", value="", lines=8)
        with gr.Row():
            add_button = gr.Button("Add to queue")
            clear_button = gr.Button("Clear queue")
        with gr.Row():
            start_button = gr.Button("Start queue", variant="primary")
            stop_button = trainer.executor.button_stop_training

        add_button.click(trainer.add, inputs=config_file, outputs=queue_box, show_progress=False)
        clear_button.click(trainer.clear, outputs=queue_box, show_progress=False)
        start_button.click(
            trainer.run_queue,
            outputs=queue_box,
            show_progress=False,
        )
        start_button.click(
            lambda: (gr.Button(visible=False), gr.Button(visible=True)),
            outputs=[start_button, stop_button],
            show_progress=False,
        )
        stop_button.click(
            trainer.executor.kill_command,
            outputs=[start_button, stop_button],
            show_progress=False,
        )
        stop_button.click(
            lambda: "",
            outputs=queue_box,
            show_progress=False,
        )

