def assign_group(name: str):
    if name.startswith("cogent3.core.alignment"):
        # c3.core.alignment.Class if possible, else just alignment
        _, _, module, *cls = name.split(".", maxsplit=4)
        if not cls:
            return "Other alignment"
        
        classes = {
            "Alignment",
            "ArrayAlignment",
            "SequenceCollection", 
            "AlignmentI",
            "_SequenceCollectionBase",
        }
        return cls[0] if cls[0] in classes else "Other alignment"
    elif name.startswith("cogent3"):
        return "Other Cogent3"
    elif name.startswith("<builtin>"):
        return "built-in"
    else:
        return "Other package"