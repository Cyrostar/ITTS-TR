import os
import json
import shutil

path_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wui_cnfg = os.path.join(path_base, "wui.json")
wui_ckpt = os.path.join(path_base, "ckpt")
wui_locs = os.path.join(path_base, "locales")
wui_outs = os.path.join(path_base, "outputs")
wui_lang = "en"

os.makedirs(wui_ckpt, exist_ok=True)
os.makedirs(wui_outs, exist_ok=True)

def load_wui():
    """Reads the global configuration from wui.json."""
    if os.path.exists(wui_cnfg):
        try:
            with open(wui_cnfg, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
    
def save_wui(last_project, language):
    """Saves global states to wui.json."""
    try:
        with open(wui_cnfg, "w") as f:
            json.dump({
                "wui_last": last_project,
                "wui_lang": language
            }, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

wui_data = load_wui()
wui_lang = wui_data.get("wui_lang", "en")
wui_last = wui_data.get("wui_last", "myproject")

prj_path = os.path.join(path_base, "projects")
pcf_path = os.path.join(prj_path, "config.json")

os.makedirs(prj_path, exist_ok=True)

project_name = ""
project_path = ""

def list_projects():
    if not os.path.exists(prj_path):
        return []
    all_items = os.listdir(prj_path)
    project_names = [
        item for item in all_items 
        if os.path.isdir(os.path.join(prj_path, item))
    ]
    return sorted(project_names)
    
def delete_project(name):
    """Safely deletes a project directory by name."""
    target = os.path.join(prj_path, name)
    if os.path.exists(target) and os.path.isdir(target):
        shutil.rmtree(target)
        return True
    return False
    
available = list_projects()

if not available:
    os.makedirs(os.path.join(prj_path, "myproject"), exist_ok=True)
    project_name = "myproject"
else:
    if wui_last in available:
        project_name = wui_last
    else:
        project_name = available[0]
        
project_path = os.path.join(prj_path, project_name)
       
save_wui(project_name, wui_lang)

def configs_directory():
    d = os.path.join(project_path, "configs")
    os.makedirs(d, exist_ok=True)
    return d
    
def models_directory():
    d = os.path.join(project_path, "models")
    os.makedirs(d, exist_ok=True)
    return d
    
def corpus_directory():
    d = os.path.join(project_path, "corpus")
    os.makedirs(d, exist_ok=True)
    return d

def tokenizer_directory():
    d = os.path.join(project_path, "tokenizers")
    os.makedirs(d, exist_ok=True)
    return d
        
def extractions_directory():
    d = os.path.join(project_path, "extractions")
    os.makedirs(d, exist_ok=True)
    return d
    
def load_guide_text(tab_name):
    
    lang = ui_language
    
    guide_path = os.path.join(path_base, "guides", tab_name, f"{lang}.md")
    
    if os.path.exists(guide_path):
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"❌ Error loading guide: {e}"
            
    fallback_path = os.path.join(path_base, "guides", "corpus", "en.md")
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
            
    return "⚠️ Documentation not found."

        
def get_available_languages():
    """Scans the locales for available language JSON files."""
    if not os.path.exists(wui_locs):
        return ["en", "tr"] # Safe fallback
    
    langs = []
    for f in os.listdir(wui_locs):
        if f.endswith(".json"):
            # Extracts 'en' from 'en_US.json' or 'tr' from 'tr_TR.json'
            lang_code = f.split("_")[0] 
            if lang_code not in langs:
                langs.append(lang_code)
                
    return sorted(langs) if langs else ["en", "tr"]
        
class Translator:
    def __init__(self):
        config = load_wui()
        # Default to "en" if not set
        self.language = config.get("wui_lang", "en")
        self.language_map = self._load_language_file(self.language)

    def _load_language_file(self, lang):
        if not os.path.exists(wui_locs):
            return {}
            
        for f in os.listdir(wui_locs):
            if f.startswith(f"{lang}_") and f.endswith(".json"):
                lang_path = os.path.join(wui_locs, f)
                try:
                    with open(lang_path, "r", encoding="utf-8") as file:
                        return json.load(file)
                except Exception as e:
                    print(f"Error loading language file {lang_path}: {e}")
                    
        return {} # Returns empty dict if file is not found

    def __call__(self, key):
        # Return translated string, fallback to key if missing
        return self.language_map.get(key, key)
        
_ = Translator() 
ui_language = _.language

my_css = """
/* --- Light Mode (Default) --- */
button[role="tab"] {
    font-size: 20px !important;
    font-weight: bold !important;
    color: #333 !important; 
}

/* --- Dark Mode --- */
.dark button[role="tab"] {
    color: #efefef !important; /* Lighter text for dark background */
}

/* --- Active Tab Style (Light Mode) --- */
button[role="tab"][aria-selected="true"] {
    color: #e44d26 !important; 
    border-color: #e44d26 !important;
}

/* --- Active Tab Style (Dark Mode - Optional tweak for contrast) --- */
.dark button[role="tab"][aria-selected="true"] {
    color: #ff5722 !important; 
    border-color: #ff5722 !important;
}

/* Increase Accordion Title Font Size */
.wui-accordion > button > span {
    font-size: 18px !important; 
    font-weight: bold !important;
}

.wui-markdown {
    padding-top: 5px !important;    /* Adjust size as needed */
    padding-bottom: 5px !important; /* Adjust size as needed */
}

.wui-button-green {
    background-color: #09845b;
}

.wui-button-blue {
    background-color: #095184;
}

.wui-button-grey {
    background-color: #3e404f;
}
"""

def language_list():

    priority = ["zh", "en", "tr", "es"]
    
    # ISO 639-1 codes for European languages
    european = [
        "es", "fr", "ru", "pt", "de", "tr", "it", "pl", "uk", "ro",
        "nl", "hu", "el", "cs", "sv", "bg", "sr", "da", "fi", "sk",
        "no", "hr", "sq", "ka", "hy", "lt", "be", "sl", "mo", "lv",
        "mk", "bs", "et", "mt", "is", "me", "ga"
    ]
    
    # ISO 639-1 codes for common Asian languages
    asian = [
        "zh", "hi", "ar", "bn", "ur", "id", "ja", "pa", "mr", "te", 
        "ta", "vi", "ko", "fa", "tl", "gu", "th", "kn", "ml", "ms", 
        "my", "uz", "ne", "az", "si", "km", "kk", "he", "tg", "tk", "lo"
    ]
    
    # ISO 639-1 codes for common Latin America languages
    americas = ["es", "fr", "pt", "nl", "ht"]
    
    # ISO 639-1 codes for common African languages
    african = [
        "sw", "ha", "yo", "am", "om", "ff", "ig", "zu", "mg", "ak", 
        "rw", "so", "xh", "af", "ln", "bm", "sn", "ny", "tw", "lg", 
        "ki", "ti", "st", "tn", "wo"
    ]
    
    # Add your language here or to priority list
    other = []
    
    # Sort the rest alphabetically for a better UI experience
    langs = list(set(european + asian + african + other))
    langs.sort()
    
    # Combine lists
    return priority + langs
    
def get_language_dict() -> dict:
    """
    Generates a dictionary mapping language codes to integer IDs.
    IDs 0, 1, and 2 are bypassed; mappings begin at 3 for 'zh'.
    """
    return {lang: index + 3 for index, lang in enumerate(language_list())}
  
def language_id(lang_code: str) -> int:
    """
    Returns the integer ID of the specified language code using the offset dictionary.
    """
    lang_map = get_language_dict()
    return lang_map.get(lang_code, 0)