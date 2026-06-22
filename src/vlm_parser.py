import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import time
from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

def check_cuda_working():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        y = x + 1
        _ = y.cpu()
        return True
    except Exception as e:
        print(f"[VLM CUDA Check] CUDA available but failed kernel execution: {e}. Falling back to CPU.")
        return False

class FlorenceParser:
    def __init__(self):
        self.device = "cuda" if check_cuda_working() else "cpu"
        self.model_id = 'microsoft/Florence-2-large' if self.device == 'cuda' else 'microsoft/Florence-2-base'
        self.processor = None
        self.model = None
        self.is_loaded = False

    def load(self):
        if self.is_loaded:
            return True
        print(f"[VLM] Loading {self.model_id} to {self.device}...")
        try:
            t0 = time.time()
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            
            # GPU Optimization: Load in FP16 on CUDA (RTX 5070 high-performance setup)
            if self.device == "cuda":
                print(f"[VLM] Loading Florence-2 in FP16 on {self.device}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                ).eval().to(self.device)
            else:
                # CPU Optimization: standard float32 (fastest on CPU)
                print("[VLM] Fallback: Loading Florence-2 in float32 on CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True
                ).eval()
                
            if hasattr(self.model, "to") and self.device != "cpu":
                self.model.to(self.device)
            print(f"[VLM] Florence-2 ready in {time.time()-t0:.1f}s")
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[VLM] Error loading Florence-2: {e}")
            return False

    def _prepare_inputs(self, task_prompt, pil_img):
        inputs = self.processor(text=task_prompt, images=pil_img, return_tensors="pt")
        if self.device == "cpu":
            model_dtype = torch.float32
        else:
            model_dtype = torch.float16
            
        inputs = {k: v.to(self.device) if k != "pixel_values" else v.to(self.device, model_dtype) for k, v in inputs.items()}
        return inputs

    def ocr_with_regions(self, pil_img: Image.Image) -> list:
        if not self.is_loaded:
            if not self.load():
                return []

        task_prompt = '<OCR_WITH_REGION>'
        inputs = self._prepare_inputs(task_prompt, pil_img)

        beams = 1 if self.device == "cpu" else 3
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=beams,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Explicit VRAM Cleanup
        del inputs, generated_ids
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(pil_img.width, pil_img.height)
        )

        res = parsed_answer.get(task_prompt, {})
        out_lines = []
        if isinstance(res, dict) and 'quad_boxes' in res and 'labels' in res:
            for bbox, label in zip(res['quad_boxes'], res['labels']):
                if len(bbox) == 8:
                    quad = [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[4], bbox[5]],
                        [bbox[6], bbox[7]]
                    ]
                else:
                    quad = [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]]
                    ]
                out_lines.append({
                    "text": label,
                    "bbox": quad,
                    "conf": 0.95
                })
        return out_lines

    def caption(self, pil_img: Image.Image) -> str:
        """Runs <DETAILED_CAPTION> to get visual style / graphics on the label."""
        if not self.is_loaded:
            if not self.load():
                return "N/A"
        
        task_prompt = '<DETAILED_CAPTION>'
        inputs = self._prepare_inputs(task_prompt, pil_img)
        
        beams = 1 if self.device == "cpu" else 3
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                early_stopping=False,
                do_sample=False,
                num_beams=beams,
            )
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Explicit VRAM Cleanup
        del inputs, generated_ids
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(pil_img.width, pil_img.height)
        )
        return parsed_answer.get(task_prompt, "").strip()

# Global singleton
florence_parser = FlorenceParser()
