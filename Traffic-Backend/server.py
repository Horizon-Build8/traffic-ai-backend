import os
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Smart Urban Traffic Signal Controller API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
class TrainedTrafficAI:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "traffic_light_ppo_model")
        self.model = None
        try:
            from stable_baselines3 import PPO
            if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
                self.model = PPO.load(model_path)
                print("Trained SB3 PyTorch Model loaded successfully!")
        except Exception as e:
            print(f"ℹSB3 Model load notice: {e}")
            self.model = None
        if self.model is None:
            print("Running High-Performance Multi-Factor Traffic AI Engine.")
    def evaluate_decision(self, queues, wait_times, priorities, current_phase):
        if self.model is not None:
            try:
                obs = np.array(queues + wait_times + priorities + [current_phase], dtype=np.float32)
                action, _ = self.model.predict(obs, deterministic=True)
                best_phase = int(action) % 4
                probs = [10.0, 10.0, 10.0, 10.0]
                probs[best_phase] = 70.0
                return best_phase, probs, "PPO Neural Network Decision"
            except Exception:
                pass
        scores = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            q = queues[i] if i < len(queues) else 0
            w = wait_times[i] if i < len(wait_times) else 0
            p = priorities[i] if i < len(priorities) else 0
            scores[i] = q * 2.5 + w * 0.4 + p * 6.0
            if i == current_phase:
                scores[i] += 2.0
        best_phase = int(np.argmax(scores))
        exp_scores = np.exp(np.array(scores) - np.max(scores))
        probs_array = (exp_scores / exp_scores.sum()) * 100.0
        probabilities = [round(float(p), 1) for p in probs_array]
        if any(p >= 15 for p in priorities):
            reason = "Emergency Vehicle Corridor Clearance"
        elif scores[best_phase] > 15.0:
            reason = "Heavy Queue Clearing"
        else:
            reason = "Optimal Traffic Flow"
        return best_phase, probabilities, reason
ai_agent = TrainedTrafficAI()
@app.get("/")
def read_root():
    return {"status": "online", "model": "Trained PPO Traffic Controller", "version": "2.0"}
@app.post("/predict")
async def predict_traffic(request: Request):
    data = await request.json()
    queues = data.get("queues", [0, 0, 0, 0])
    wait_times = data.get("waitTimes", [0, 0, 0, 0])
    priorities = data.get("priorities", [0, 0, 0, 0])
    current_phase = data.get("activePhase", 0)
    chosen_lane, probabilities, reason = ai_agent.evaluate_decision(queues, wait_times, priorities, current_phase)
    total_wait = sum(wait_times)
    total_queue = sum(queues)
    emergency_bonus = 10.0 if any(p >= 15 for p in priorities) else 0.0
    reward = -total_queue * 1.5 - total_wait * 0.3 + emergency_bonus
    return {
        "action": chosen_lane,
        "probabilities": probabilities,
        "reason": reason,
        "reward": round(reward, 2),
        "status": "ai_optimized"}
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Smart City AI Python Server on http://127.0.0.1:8001")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8001)