import os
import tempfile
import pytest
from glue.core.persistence import DynamicComponentStore, ComponentSpec, ComponentType
from datetime import datetime
import asyncio
import json

@pytest.mark.asyncio
async def test_component_store_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DynamicComponentStore(storage_dir=tmpdir)
        comp_id = await store.save_component(
            component_type=ComponentType.TOOL,
            name="TestTool",
            code="print('hi')",
            metadata={"foo": "bar"},
        )
        loaded = await store.load_component(comp_id)
        assert loaded.name == "TestTool"
        assert loaded.metadata["foo"] == "bar"

@pytest.mark.asyncio
async def test_component_store_list_and_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DynamicComponentStore(storage_dir=tmpdir)
        id1 = await store.save_component(
            component_type=ComponentType.TOOL,
            name="Tool1",
            code="print('1')",
            metadata={},
        )
        id2 = await store.save_component(
            component_type=ComponentType.MCP_SERVER, # Changed type for variety
            name="Server2", # Changed name
            code="print('2')",
            metadata={},
        )
        # Test listing all components
        all_comps = await store.list_components()
        all_ids = {c.id for c in all_comps}
        assert {id1, id2} <= all_ids
        assert len(all_comps) >= 2 # Check at least these two are present

        # Test listing specific type
        tool_comps = await store.list_components(ComponentType.TOOL)
        tool_ids = {c.id for c in tool_comps}
        assert id1 in tool_ids
        assert id2 not in tool_ids # Server2 should not be in TOOL list

        server_comps = await store.list_components(ComponentType.MCP_SERVER)
        server_ids = {c.id for c in server_comps}
        assert id2 in server_ids
        assert id1 not in server_ids # Tool1 should not be in MCP_SERVER list
        
        await store.delete_component(id1)
        
        # Verify deletion when listing all
        all_comps_after_delete = await store.list_components()
        all_ids_after_delete = {c.id for c in all_comps_after_delete}
        assert id2 in all_ids_after_delete and id1 not in all_ids_after_delete

        # Verify deletion when listing specific type
        tool_comps_after_delete = await store.list_components(ComponentType.TOOL)
        assert id1 not in {c.id for c in tool_comps_after_delete}

@pytest.mark.asyncio
async def test_component_store_update_component():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DynamicComponentStore(storage_dir=tmpdir)
        comp_id = await store.save_component(
            component_type=ComponentType.TOOL,
            name="OriginalTool",
            code="print('original')",
            metadata={"version": 1},
        )
        original_spec = await store.load_component(comp_id)
        original_updated_at = original_spec.updated_at

        # Update code and metadata
        await asyncio.sleep(0.01) # Ensure timestamp changes
        await store.update_component(
            comp_id,
            code="print('updated')",
            metadata={"version": 2, "author": "test"},
        )
        updated_spec = await store.load_component(comp_id)
        assert updated_spec.name == "OriginalTool" # Name should not change
        assert updated_spec.code == "print('updated')"
        assert updated_spec.metadata["version"] == 2
        assert updated_spec.metadata["author"] == "test"
        assert updated_spec.updated_at > original_updated_at

        # Update only code
        original_updated_at = updated_spec.updated_at
        await asyncio.sleep(0.01) # Ensure timestamp changes
        await store.update_component(comp_id, code="print('final')")
        final_spec = await store.load_component(comp_id)
        assert final_spec.code == "print('final')"
        assert final_spec.metadata["version"] == 2 # Metadata should persist
        assert final_spec.updated_at > original_updated_at

        # Update only metadata
        original_updated_at = final_spec.updated_at
        await asyncio.sleep(0.01) # Ensure timestamp changes
        await store.update_component(comp_id, metadata={"status": "active"})
        final_meta_spec = await store.load_component(comp_id)
        assert final_meta_spec.code == "print('final')" # Code should persist
        assert final_meta_spec.metadata["status"] == "active"
        assert final_meta_spec.updated_at > original_updated_at

@pytest.mark.asyncio
async def test_component_store_load_or_delete_non_existent():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DynamicComponentStore(storage_dir=tmpdir)
        non_existent_id = "non-existent-id"
        
        with pytest.raises(FileNotFoundError):
            await store.load_component(non_existent_id)
            
        with pytest.raises(FileNotFoundError):
            await store.delete_component(non_existent_id)

@pytest.mark.asyncio
async def test_component_store_default_storage_dir():
    # This test relies on the default directory creation logic
    # It's a bit harder to test in isolation without mocking os.path.expanduser or os.makedirs
    # For now, we'll just instantiate it and check if the dir is set
    store = DynamicComponentStore()
    expected_dir = os.path.join(os.path.expanduser("~"), ".glue", "components")
    assert store.storage_dir == expected_dir
    # We can also check if the subdirectories for component types are created
    for component_type in ComponentType:
        assert os.path.isdir(os.path.join(store.storage_dir, component_type.value))
    # Basic save and load to ensure it works with default path
    comp_id = await store.save_component(ComponentType.TOOL, "DefaultDirTool", "code", {})
    assert comp_id is not None
    loaded = await store.load_component(comp_id)
    assert loaded.name == "DefaultDirTool"
    await store.delete_component(comp_id) # Clean up

@pytest.mark.asyncio
async def test_list_components_empty_or_specific_type_no_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DynamicComponentStore(storage_dir=tmpdir)
        
        # List all when empty
        all_empty = await store.list_components()
        assert len(all_empty) == 0
        
        # List specific type when empty
        tool_empty = await store.list_components(ComponentType.TOOL)
        assert len(tool_empty) == 0
        
        # Add a component of one type
        await store.save_component(ComponentType.MCP_SERVER, "Server1", "code", {})
        
        # List a different type, should be empty
        tool_still_empty = await store.list_components(ComponentType.TOOL)
        assert len(tool_still_empty) == 0

def test_get_component_path_logic(tmpdir): # Use pytest's tmpdir fixture for sync test
    store = DynamicComponentStore(storage_dir=str(tmpdir))
    comp_id = "test-id"
    
    # Test with component_type provided
    path_with_type = store._get_component_path(comp_id, ComponentType.TOOL)
    expected_path_with_type = os.path.join(str(tmpdir), ComponentType.TOOL.value, f"{comp_id}.json")
    assert path_with_type == expected_path_with_type
    
    # Test without component_type, component doesn't exist (should default to "unknown")
    # _get_component_path will search, and if not found, construct a path in "unknown"
    # This behavior is a bit implicit, the "unknown" dir is not created by _ensure_storage_dir
    # Let's simulate a component existing to test the search logic
    
    # Create a dummy file for an existing component
    tool_dir = os.path.join(str(tmpdir), ComponentType.TOOL.value)
    os.makedirs(tool_dir, exist_ok=True)
    existing_comp_id = "existing-tool-id"
    existing_file_path = os.path.join(tool_dir, f"{existing_comp_id}.json")
    with open(existing_file_path, "w") as f:
        json.dump({"id": existing_comp_id, "component_type": "tool", "name": "Test", "code": "", "created_at": "", "updated_at": ""}, f)
        
    path_found = store._get_component_path(existing_comp_id)
    assert path_found == existing_file_path
    
    # Test for a non-existent component ID, without type (should go to unknown)
    # The method _get_component_path itself doesn't create the "unknown" directory.
    # It just constructs the path. The actual file operation (like open) would fail if it doesn't exist.
    # The current implementation of _get_component_path when a component is not found and no type is given
    # will return a path within an "unknown" subdirectory. This is fine for constructing paths for new components.
    non_existent_path = store._get_component_path("another-new-id")
    expected_non_existent_path = os.path.join(str(tmpdir), "unknown", "another-new-id.json")
    assert non_existent_path == expected_non_existent_path

@pytest.mark.asyncio
async def test_component_spec_from_dict():
    now_iso = datetime.now().isoformat()
    data = {
        "id": "spec-id",
        "component_type": "tool", # string value
        "name": "SpecTool",
        "code": "pass",
        "created_at": now_iso,
        "updated_at": now_iso,
        "metadata": {"key": "value"}
    }
    spec = ComponentSpec.from_dict(data)
    assert spec.id == "spec-id"
    assert spec.component_type == ComponentType.TOOL # Should be enum
    assert spec.name == "SpecTool"
    assert spec.metadata["key"] == "value"

    data_with_enum = {
        "id": "spec-id-2",
        "component_type": ComponentType.MCP_SERVER, # enum value
        "name": "SpecServer",
        "code": "pass",
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    spec_enum = ComponentSpec.from_dict(data_with_enum)
    assert spec_enum.component_type == ComponentType.MCP_SERVER
