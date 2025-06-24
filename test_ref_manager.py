import json
import os
from pathlib import Path
import torch

class ReferenceManager:
    def __init__(self):
        self.store_reference = os.environ.get('STORE_REFERENCE', 'false').lower() == 'true'
        self.reference_dir = None
        self.reference_file = None

    def setup_reference_dir(self, test_file_path: str):
        self.reference_dir = Path(test_file_path).parent / "reference"
        os.makedirs(self.reference_dir, exist_ok=True)
        # Use the test file name as the reference file name
        test_file_name = Path(test_file_path).stem
        self.reference_file = self.reference_dir / f"{test_file_name}.json"

    def compare(self, data: torch.Tensor, key: str, atol: float = 1e-6, rtol: float = 1e-6):
        if self.store_reference:
            self._store_reference_data(key, data)
        else:
            ref_data = self._load_reference_data(key)
            assert torch.allclose(data, ref_data, atol=atol, rtol=rtol)

    def _format_nested_list(self, obj, indent_level=0):
        """Custom formatter to keep innermost arrays on single lines."""
        indent = " " * indent_level
        
        if isinstance(obj, list):
            # Check if this is an innermost array (contains only numbers)
            if all(isinstance(item, (int, float)) for item in obj):
                return "[ " + ", ".join(str(item) for item in obj) + " ]"
            else:
                # This is a nested structure
                items = []
                for item in obj:
                    formatted_item = self._format_nested_list(item, indent_level + 1)
                    items.append(f"{indent} {formatted_item}")
                return "[\n" + ",\n".join(items) + f"\n{indent}]"
        else:
            return json.dumps(obj)

    def _load_existing_references(self):
        """Load existing reference data from file."""
        if self.reference_file.exists():
            with open(self.reference_file, 'r') as f:
                return json.load(f)
        return {}

    def _store_reference_data(self, key: str, data: torch.Tensor):
        """Store reference tensor data as JSON with the given key."""
        if self.reference_dir is None:
            raise ValueError("Reference directory not set. Call setup_reference_dir first.")
        if self.store_reference:
            # Load existing references
            all_references = self._load_existing_references()
            
            # Add new reference data
            data_dict = {
                'data': data.tolist(),
                'shape': list(data.shape),
                'dtype': str(data.dtype)
            }
            all_references[key] = data_dict
            
            # Format the entire JSON with custom formatting
            json_content = "{\n"
            items = []
            for ref_key, ref_data in all_references.items():
                formatted_data = self._format_nested_list(ref_data['data'], 2)
                item_content = f' "{ref_key}": {{\n'
                item_content += f'  "data": {formatted_data},\n'
                item_content += f'  "shape": {json.dumps(ref_data["shape"])},\n'
                item_content += f'  "dtype": {json.dumps(ref_data["dtype"])}\n'
                item_content += ' }'
                items.append(item_content)
            
            json_content += ",\n".join(items)
            json_content += "\n}"
            
            with open(self.reference_file, 'w') as f:
                f.write(json_content)
            print(f"Stored reference data '{key}' to {self.reference_file}")

    def _load_reference_data(self, key: str) -> torch.Tensor:
        """Load reference tensor data from JSON using the given key."""
        if self.reference_dir is None:
            raise ValueError("Reference directory not set. Call setup_reference_dir first.")
        if not self.reference_file.exists():
            raise FileNotFoundError(f"Reference file not found: {self.reference_file}")
        
        with open(self.reference_file, 'r') as f:
            all_references = json.load(f)
        
        if key not in all_references:
            raise KeyError(f"Reference key '{key}' not found in {self.reference_file}")
        
        data_dict = all_references[key]
        
        dtype_map = {
            'torch.int32': torch.int32,
            'torch.float32': torch.float32,
            'torch.float64': torch.float64,
            'torch.int64': torch.int64,
        }
        dtype = dtype_map.get(data_dict['dtype'], torch.float32)
        tensor = torch.tensor(data_dict['data'], dtype=dtype)
        
        return tensor