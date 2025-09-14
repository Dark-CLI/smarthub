import pytest
from ha.client import HAClient

@pytest.mark.asyncio
async def test_states_and_services_live():
    ha = HAClient()

    # States
    states = await ha.states()
    assert isinstance(states, list)
    assert len(states) > 0

    # Services (LIST of domain blocks)
    services = await ha.services()
    assert isinstance(services, list)
    assert len(services) > 0

    # Check that at least one common domain exists
    domains = {block.get("domain") for block in services if isinstance(block, dict)}
    assert ("light" in domains) or ("switch" in domains)

@pytest.mark.asyncio
async def test_call_light_service_live():
    ha = HAClient()

    # ⚠️ Change this to a real entity_id in your HA
    entity_id = "light.devices_balcony_light"

    # Turn on
    result_on = await ha.call_service("light", "turn_on", entity_id=entity_id)
    assert result_on is None or isinstance(result_on, (dict, list))
    print("Turned on:", entity_id)

    # Turn off
    result_off = await ha.call_service("light", "turn_off", entity_id=entity_id)
    assert result_off is None or isinstance(result_off, (dict, list))
    print("Turned off:", entity_id)


