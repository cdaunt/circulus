import asyncio
import websockets
import jax
import jax.numpy as jnp
import numpy as np
import json

# --- 1. JAX Kernel ---
# We add frequency and amplitude as arguments.
# JIT compilation handles scalar changes very efficiently.
# Change this line:
# @jax.jit 

# To this:
@jax.jit(static_argnums=(1,)) 
def compute_waveform(t, points, freq, amp):
    # ... rest of function remains the same
    x = jnp.linspace(0, 10, points)
    y = jnp.sin(x * freq + t * 5.0) * amp
    y += 0.2 * jnp.sin(x * (freq * 2.5) + t * 2.0)
    return y.astype(jnp.float32)

# Global State to share between the Listener and Sender
state = {
    "freq": 2.0,
    "amp": 1.0,
    "running": True
}

async def consumer_handler(websocket):
    """Listens for JSON messages from the browser to update state."""
    async for message in websocket:
        try:
            data = json.loads(message)
            if 'freq' in data:
                state['freq'] = float(data['freq'])
            if 'amp' in data:
                state['amp'] = float(data['amp'])
            print(f"Updated: {state}")
        except json.JSONDecodeError:
            pass

async def producer_handler(websocket):
    """Continuously runs JAX and streams binary data to browser."""
    t = 0.0
    points = 10000
    
    while state["running"]:
        # 1. Run JAX with CURRENT state values
        # Note: We pass standard python floats, JAX handles the casting
        waveform_jax = compute_waveform(t, points, state['freq'], state['amp'])
        
        # 2. Serialize
        data = np.array(waveform_jax, dtype=np.float32).tobytes()
        
        # 3. Send
        try:
            await websocket.send(data)
        except websockets.exceptions.ConnectionClosed:
            break
            
        t += 0.016
        await asyncio.sleep(0.016) # ~60 FPS cap

async def handler(websocket):
    # Run both tasks concurrently. 
    # If the connection drops, one will fail/finish, and we cancel the other.
    consumer_task = asyncio.create_task(consumer_handler(websocket))
    producer_task = asyncio.create_task(producer_handler(websocket))
    
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    
    for task in pending:
        task.cancel()

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Bi-directional JAX Server running on localhost:8765")
        await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())