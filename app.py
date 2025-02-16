import io
import random
import logging
import sys
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info(f"Python version: {sys.version}")
logger.info(f"Torch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

app = Flask(__name__)

# --- Choose which model implementation to use ---
# For Apple Silicon using MLX VLM with SmolLM-135M-Instruct-4bit:
USE_MLXVLM = True

if USE_MLXVLM:
    logger.info("Loading MLX VLM model for Apple Silicon...")
    try:
        # Import MLX VLM functions
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        # Specify your model identifier.
        model_path = "mlx-community/SmolVLM-256M-Instruct-bf16"
        model, processor = load(model_path)
        config = load_config(model_path)
        logger.info("MLX VLM loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading MLX VLM: {e}")
        raise
else:
    # Fallback to another model branch (for example, your existing SmolVLM code)
    from transformers import AutoProcessor, AutoModelForVision2Seq
    logger.info("Loading SmolVLM model and processor...")
    try:
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.bfloat16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logger.info("SmolVLM loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading SmolVLM: {e}")
        raise

# --- Define Ad structures and catalogs (unchanged) ---
class Ad:
    def __init__(self, title, description, price, rating, image_size=(400, 300)):
        self.title = title
        self.description = description
        self.price = price
        self.rating = rating
        self.image_placeholder = "https://blog.udemy.com/wp-content/uploads/2014/05/bigstock-Vector-Promotion-Concept-Fla-57726575.jpg"

def format_stars(rating):
    full_stars = int(rating)
    has_half = rating - full_stars >= 0.5
    stars = "★" * full_stars
    if has_half:
        stars += "½"
    return stars

def format_ad_html(ad):
    return f"""
    <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: white;">
        <img src="{ad.image_placeholder}" alt="{ad.title}" style="width: 300px; height: auto; border-radius: 4px;">
        <h3 style="margin: 10px 0; color: #333; font-size: 18px;">{ad.title}</h3>
        <p style="color: #666; margin: 8px 0;">{ad.description}</p>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
            <span style="color: #2ecc71; font-weight: bold; font-size: 20px;">{ad.price}</span>
            <span style="color: #f1c40f; font-size: 16px;">{format_stars(ad.rating)} ({ad.rating})</span>
        </div>
    </div>
    """

# Ad catalogs remain unchanged…
ads_age = {
    "20-30": [
        Ad("Ultra Boost X Sneakers", "Limited edition sneakers with cutting-edge comfort technology", "$199.99", 4.8),
        Ad("Smart Fitness Watch Pro", "Track your workouts, sleep, and lifestyle with AI-powered insights", "$299.99", 4.7),
        Ad("Urban Streetwear Collection", "Express yourself with our latest street-inspired fashion drops", "$149.99", 4.6)
    ],
    "30-40": [
        Ad("Premium Coffee Maker Plus", "Barista-quality coffee at home with smart temperature control", "$399.99", 4.9),
        Ad("Professional Series Smartwatch", "Elegant timepiece with advanced productivity features", "$449.99", 4.8),
        Ad("Home Automation Starter Kit", "Transform your space with intelligent lighting and security", "$299.99", 4.7)
    ],
    "40-50": [
        Ad("Artisanal Wine Selection", "Curated collection of premium wines from renowned vineyards", "$599.99", 4.9),
        Ad("Luxury Watch Collection", "Timeless elegance meets modern craftsmanship", "$1999.99", 4.8),
        Ad("Executive Wardrobe Essentials", "Premium suits and accessories for the distinguished professional", "$899.99", 4.7)
    ],
    "50+": [
        Ad("Wellness Supplement Bundle", "Comprehensive nutrition support for active aging", "$129.99", 4.8),
        Ad("Comfort-Tech Footwear", "Advanced ergonomic design for all-day comfort", "$159.99", 4.7),
        Ad("Luxury Travel Experiences", "Curated adventures with premium accommodations and service", "$2999.99", 4.9)
    ]
}

ads_gender = {
    "Male": [
        Ad("Premium Grooming Kit", "Complete care collection with precision trimmer and luxurious skincare", "$199.99", 4.7),
        Ad("Modern Menswear Essentials", "Versatile pieces for the contemporary gentleman", "$299.99", 4.6),
        Ad("Sports Performance Collection", "Advanced gear for your active lifestyle", "$249.99", 4.8)
    ],
    "Female": [
        Ad("Luxury Beauty Collection", "Premium skincare and makeup for radiant beauty", "$299.99", 4.9),
        Ad("Designer Handbag Selection", "Exclusive bags from world-renowned fashion houses", "$999.99", 4.8),
        Ad("Jewelry & Accessories Edit", "Timeless pieces to elevate every outfit", "$499.99", 4.7)
    ]
}

ads_mood = {
    "Happy": [
        Ad("Party Planning Bundle", "Everything you need for your next celebration", "$199.99", 4.7),
        Ad("Adventure Gear Set", "Premium outdoor equipment for your next expedition", "$399.99", 4.8),
        Ad("Entertainment Package Plus", "Stream, game, and enjoy with our premium entertainment system", "$599.99", 4.6)
    ],
    "Neutral": [
        Ad("Home Essentials Collection", "Quality basics for everyday living", "$249.99", 4.5),
        Ad("Smart Kitchen Appliances", "Efficient cooking with innovative technology", "$699.99", 4.7),
        Ad("Casual Dining Experience", "Discover local restaurants with exclusive offers", "$99.99", 4.6)
    ],
    "Sad": [
        Ad("Wellness Retreat Package", "Rejuvenating spa experiences for mind and body", "$399.99", 4.9),
        Ad("Comfort Food Delivery", "Gourmet comfort meals delivered to your door", "$79.99", 4.7),
        Ad("Self-Care Collection", "Premium products for relaxation and wellness", "$199.99", 4.8)
    ]
}

ads_style = {
    "Sporty": [
        Ad("Pro Athletic Wear", "High-performance gear for serious athletes", "$199.99", 4.8),
        Ad("Premium Running Shoes", "Advanced cushioning and support for every run", "$179.99", 4.7),
        Ad("Sports Tech Bundle", "Track and improve your performance with smart devices", "$299.99", 4.6)
    ],
    "Casual": [
        Ad("Essential Comfort Collection", "Effortless style for everyday wear", "$149.99", 4.5),
        Ad("Lifestyle Sneaker Edit", "Trendy and comfortable footwear for any occasion", "$129.99", 4.6),
        Ad("Casual Basics Bundle", "Build your perfect everyday wardrobe", "$199.99", 4.7)
    ],
    "Formal": [
        Ad("Luxury Suit Collection", "Bespoke tailoring with premium fabrics", "$999.99", 4.9),
        Ad("Executive Accessories", "Fine watches and leather goods for professionals", "$499.99", 4.8),
        Ad("Premium Business Wear", "Sophisticated attire for the modern executive", "$799.99", 4.7)
    ],
    "Vintage": [
        Ad("Classic Collection Pieces", "Timeless fashion with a modern twist", "$299.99", 4.7),
        Ad("Retro-Inspired Accessories", "Vintage-style pieces for unique charm", "$199.99", 4.6),
        Ad("Heritage Fashion Edit", "Contemporary takes on classic designs", "$399.99", 4.8)
    ]
}

# --- Utility functions ---
def extract_answer(response):
    """Extract a clean answer from the model output."""
    # Remove the end_of_utterance tag
    response = response.replace("<end_of_utterance>", "")
    
    # Remove any leading/trailing whitespace and periods
    response = response.strip(". ")
    
    # You might also want to add additional cleaning if needed:
    # - Convert to title case to match your dictionary keys
    response = response.title()
    
    return response

# --- MLX VLM segmentation function ---
def ask_mlx_vlm(image, question):
    """
    Given a PIL image and a question, use MLX VLM to generate an answer.
    Note: MLX VLM's generate function expects a list of images.
    """
    logger.info(f"Processing question: {question}")
    try:
        formatted_prompt = apply_chat_template(processor, config, question, num_images=1)
        # Pass the image inside a list. Depending on MLX VLM's requirements, you may need to convert the PIL image.
        output = generate(model, processor, formatted_prompt, [image], verbose=False)
        logger.info(f"Generated response: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in ask_mlx_vlm: {e}")
        return f"Error processing question: {str(e)}"

# --- Segmentation and Ad Selection ---

@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files["image"]
        image = Image.open(image_file.stream).convert("RGB")
        logger.info(f"Received image of size: {image.size}")
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return jsonify({"error": str(e)}), 500

    segmentation = {}
    questions = {
        "age": "Based on the person's facial features, what is the age range? Answer with one of: '20-30', '30-40', '40-50', '50+'.",
        "gender": "Based on the person's appearance, what is their likely gender? Answer with 'Male' or 'Female'.",
        "mood": "How would you describe the person's mood based on their expression? Answer with 'Happy', 'Neutral', or 'Sad'.",
        "style": "Based on the person's style, what fashion category do they belong to? Answer with 'Sporty', 'Casual', 'Formal', or 'Vintage'."
    }

    for key, question in questions.items():
        logger.info(f"Segmenting for {key} using question: {question}")
        try:
            response = ask_mlx_vlm(image, question)
            segmentation[key] = extract_answer(response)
            logger.info(f"Segmentation result for {key}: {segmentation[key]}")
        except Exception as e:
            logger.error(f"Error during segmentation for {key}: {e}")
            segmentation[key] = "N/A"

    # Select one random ad from each category based on segmentation results
    selected_ads = {
        'age': random.choice(ads_age.get(segmentation.get("age"), [Ad("Default Age Ad", "Personalized recommendations for you", "$0.00", 5.0)])),
        'gender': random.choice(ads_gender.get(segmentation.get("gender"), [Ad("Default Gender Ad", "Curated selections for you", "$0.00", 5.0)])),
        'mood': random.choice(ads_mood.get(segmentation.get("mood"), [Ad("Default Mood Ad", "Special picks for your mood", "$0.00", 5.0)])),
        'style': random.choice(ads_style.get(segmentation.get("style"), [Ad("Default Style Ad", "Trending items for your style", "$0.00", 5.0)]))
    }

    # Format individual ad cards
    def format_ad_html(ad):
        return f"""
        <div class="ad-card">
            <img src="{ad.image_placeholder}" alt="{ad.title}" class="ad-image">
            <div class="ad-content">
                <h3 class="ad-title">{ad.title}</h3>
                <p class="ad-description">{ad.description}</p>
                <div class="ad-footer">
                    <span class="ad-price">{ad.price}</span>
                    <span class="ad-rating">{format_stars(ad.rating)} ({ad.rating})</span>
                </div>
            </div>
        </div>
        """

    html_output = f"""
    <style>
        .results-container {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .segmentation-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .segmentation-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        
        .segmentation-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .ads-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }}
        
        .ad-card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .ad-card:hover {{
            transform: translateY(-5px);
        }}
        
        .ad-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
        }}
        
        .ad-content {{
            padding: 15px;
        }}
        
        .ad-title {{
            margin: 0 0 10px 0;
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }}
        
        .ad-description {{
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            line-height: 1.4;
        }}
        
        .ad-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .ad-price {{
            color: #2ecc71;
            font-weight: bold;
            font-size: 18px;
        }}
        
        .ad-rating {{
            color: #f1c40f;
            font-size: 14px;
        }}
        
        @media (max-width: 1024px) {{
            .ads-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        @media (max-width: 640px) {{
            .ads-grid {{
                grid-template-columns: 1fr;
            }}
            .segmentation-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>

    <div class="results-container">
        <div class="segmentation-card">
            <div class="segmentation-grid">
                <div class="segmentation-item">
                    <span>🎯</span>
                    <span>Age: {segmentation.get('age', 'N/A')}</span>
                </div>
                <div class="segmentation-item">
                    <span>👤</span>
                    <span>Gender: {segmentation.get('gender', 'N/A')}</span>
                </div>
                <div class="segmentation-item">
                    <span>😊</span>
                    <span>Mood: {segmentation.get('mood', 'N/A')}</span>
                </div>
                <div class="segmentation-item">
                    <span>👔</span>
                    <span>Style: {segmentation.get('style', 'N/A')}</span>
                </div>
            </div>
        </div>
        
        <div class="ads-grid">
            {format_ad_html(selected_ads['age'])}
            {format_ad_html(selected_ads['gender'])}
            {format_ad_html(selected_ads['mood'])}
            {format_ad_html(selected_ads['style'])}
        </div>
    </div>
    """
    return jsonify({"html": html_output})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
