import httpx, time

system = """
You are the single-turn decision & reply layer for a smart-home assistant.
Return EXACTLY ONE LINE JSON (no prose). One of:

{"mode":"EXECUTE","device":"<entity_id>","action":"<domain.service>","args":{},"reply":"<short confirmation>"}
{"mode":"REPLY","text":"<short answer>"}

Inputs:
- user_message, context, recent, keywords
- devices: [{entity_id, name, domain, area}]
- actions: [{action, domain, service, fields}]
Output ONE line of JSON only.
"""
user = """
'user_message="turn on the light"\ncontext={"room":"kitchen"}\ndevices=[["light.devices_kitchen_strip_light",{"entity_id":"light.devices_kitchen_strip_light","state":"off","attributes":{"supported_color_modes":["onoff"],"color_mode":null,"friendly_name":"Devices Kitchen Strip Light","supported_features":8}}],["light.devices_dimmer_lamp",{"entity_id":"light.devices_dimmer_lamp","state":"off","attributes":{"supported_color_modes":["brightness"],"color_mode":null,"brightness":null,"friendly_name":"Devices Dimmer Lamp","supported_features":40}}],["button.devices_reboot_system",{"entity_id":"button.devices_reboot_system","state":"unknown","attributes":{"device_class":"restart","icon":"mdi:restart","friendly_name":"Devices Reboot System"}}],["light.devices_office_desk_lamp",{"entity_id":"light.devices_office_desk_lamp","state":"off","attributes":{"supported_color_modes":["onoff"],"color_mode":null,"friendly_name":"Devices Office Desk Lamp","supported_features":8}}],["button.devices_toggle_scene_movie_mode",{"entity_id":"button.devices_toggle_scene_movie_mode","state":"unknown","attributes":{"friendly_name":"Devices Toggle Scene: Movie Mode"}}],["light.devices_bathroom_mirror_light",{"entity_id":"light.devices_bathroom_mirror_light","state":"on","attributes":{"supported_color_modes":["onoff"],"color_mode":"onoff","friendly_name":"Devices Bathroom Mirror Light","supported_features":8}}]]\nactions=[{"key":"service:switch.turn_on","action":"switch.turn_on","domain":"switch","service":"turn_on","fields":[]},{"key":"service:light.turn_off","action":"light.turn_off","domain":"light","service":"turn_off","fields":["transition","advanced_fields"]},{"key":"service:input_boolean.turn_on","action":"input_boolean.turn_on","domain":"input_boolean","service":"turn_on","fields":[]},{"key":"service:light.turn_on","action":"light.turn_on","domain":"light","service":"turn_on","fields":["transition","rgb_color","color_temp_kelvin","brightness_pct","brightness_step_pct","effect","advanced_fields"]},{"key":"service:homeassistant.turn_on","action":"homeassistant.turn_on","domain":"homeassistant","service":"turn_on","fields":[]},{"key":"service:script.turn_on","action":"script.turn_on","domain":"script","service":"turn_on","fields":[]},{"action":"button.press","domain":"button","service":"press","fields":[],"description":"Press the button entity."},{"action":"light.turn_on","domain":"light","service":"turn_on","fields":["transition","rgb_color","color_temp_kelvin","brightness_pct","brightness_step_pct","effect","advanced_fields"],"description":"Turns on one or more lights and adjusts their properties, even when they are turned on already."},{"action":"light.turn_off","domain":"light","service":"turn_off","fields":["transition","advanced_fields"],"description":"Turns off one or more lights."},{"action":"light.toggle","domain":"light","service":"toggle","fields":["transition","rgb_color","color_temp_kelvin","brightness_pct","effect","advanced_fields"],"description":"Toggles one or more lights, from on to off, or off to on, based on their current state."}]'
"""

t0 = time.perf_counter()
r = httpx.post("http://localhost:11434/api/generate", json={
    "model": "llama3.1:latest",
    "system": system,
    "prompt": user,
    "num_ctx": 4096,
    "stream": False
})
print(r.json()["response"])
print(f"Time: {(time.perf_counter() - t0)*1000:.1f} ms")
