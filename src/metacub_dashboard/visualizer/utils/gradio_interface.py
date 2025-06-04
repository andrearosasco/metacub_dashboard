
import time
import gradio as gr
from gradio_rerun import Rerun

from gradio_rerun.events import (
    TimelineChange,
    TimeUpdate,
)

def build_gradio_interface(
    rec,
    stop_streaming_event,
):
    """
    Build the Gradio interface for the Rerun viewer.
    """

    # Define Callbacks

    def track_current_time(evt: TimeUpdate):
        return evt.payload.time

    def track_current_timeline_and_time(evt: TimelineChange):
        return evt.payload.timeline, evt.payload.time

    def initialize():
        stream_to_gradio = rec.binary_stream()
        while not stop_streaming_event.is_set():
            data_chunk = stream_to_gradio.read()
            if data_chunk:
                yield data_chunk
            else:
                time.sleep(0.001)
            if stop_streaming_event.is_set() and not data_chunk:
                break

    # Create the Gradio interface

    with gr.Blocks() as demo:
        with gr.Tab("Streaming"):
            with gr.Row():
                viewer = Rerun(
                    streaming=True,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "hidden",
                        "selection": "hidden",
                    },
                )

            # Also store the current timeline and time of the viewer in the session state.
            current_timeline = gr.State("")
            current_time = gr.State(0.0)

            viewer.time_update(track_current_time, outputs=[current_time])
            viewer.timeline_change(
                track_current_timeline_and_time, outputs=[current_timeline, current_time]
            )
        demo.load(            
            fn=initialize, 
            inputs=[],
            outputs=[viewer],)
        
    return demo