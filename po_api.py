from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Union, Any
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import redis
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import signal
import sys
import time
from itertools import cycle
from fuzzywuzzy import process, fuzz
from enum import Enum
from pydantic import ValidationError

# Configure for Azure App Service
PORT = int(os.getenv('PORT', 8000))
HOST = os.getenv('HOST', '0.0.0.0')

# Initialize FastAPI App
app = FastAPI(
    title="ProcureAI API",
    description="API for creating and managing purchase orders",
    version="1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis Configuration
try:
    redis_client = redis.Redis(
        host='social-rabbit-56166.upstash.io',
        port=6379,
        password='AdtmAAIjcDE5MzE4YzQ2NDcyY2E0ZTI3YTc3ZmNiYzBkYzBlYzk5YnAxMA',
        ssl=True,
        decode_responses=True
    )
    # Test Redis connection
    redis_client.ping()
    logger.info("Redis connection established successfully")
    REDIS_ENABLED = True
except ImportError:
    logger.warning("Redis package not installed, continuing without caching")
    REDIS_ENABLED = False
    redis_client = None
except redis.ConnectionError as e:
    logger.warning(f"Redis connection failed: {str(e)}, continuing without caching")
    REDIS_ENABLED = False
    redis_client = None
except Exception as e:
    logger.warning(f"Redis error: {str(e)}, continuing without caching")
    REDIS_ENABLED = False
    redis_client = None

# Update path to use environment variable or default
BASE_PATH = os.getenv('PO_DATA_PATH', os.path.dirname(os.path.realpath(__file__)))
PO_DATA_FILE = os.path.join(BASE_PATH, os.getenv('PO_DATA_FILE', 'PO Data_formatted.xlsx'))

# Define product categories first
class ProductCategory(str, Enum):
    MEAT = "Meat & Poultry"
    SEAFOOD = "Fish & Seafood"
    DAIRY = "Dairy & Eggs"
    PRODUCE = "Fruits & Vegetables"

# ML Model class
class POPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False
        self.le_vendor = LabelEncoder()
        self.le_item = LabelEncoder()

    def train(self, df: pd.DataFrame):
        """Train the ML model on historical data"""
        try:
            # Convert BillDate to datetime
            df['BillDate'] = pd.to_datetime(df['BillDate'])
            
            # Prepare features
            self.le_vendor.fit(df['VendorID'].unique())
            self.le_item.fit(df['ItemName'].unique())
            
            features = np.array([
                self.le_vendor.transform(df['VendorID']),
                self.le_item.transform(df['ItemName']),
                df['BillDate'].dt.dayofweek,
                df['BillDate'].dt.month,
                df['BillDate'].dt.day,
                df['Quantity'].astype(float)
            ]).T
            
            target = df['Quantity'].astype(float)
            
            # Train model
            self.model.fit(features, target)
            self.trained = True
            logger.info("Model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict_stock_level(self, vendor_id: str, item_name: str, current_stock: float) -> dict:
        """Predict future stock levels for an item"""
        try:
            today = datetime.now()
            
            features = np.array([[
                self.le_vendor.transform([vendor_id])[0],
                self.le_item.transform([item_name])[0],
                today.weekday(),
                today.month,
                today.day,
                float(current_stock)
            ]])
            
            predicted_usage = float(self.model.predict(features)[0])
            predicted_stock = max(0, current_stock - predicted_usage)
            
            return {
                "current_stock": float(current_stock),
                "predicted_stock": float(predicted_stock),
                "predicted_usage": float(predicted_usage),
                "needs_reorder": True if predicted_stock < 10 else False,
                "recommended_order": float(max(20 - predicted_stock, 0)),
                "confidence_score": 0.85
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    async def get_recommendations(self) -> List[dict]:
        """Get ML-based recommendations for POs across all vendors"""
        try:
            df = load_data()
            if df.empty:
                return []
            
            # Train model if not trained
            if not self.trained:
                self.train(df)
            
            # Get current inventory levels
            inventory = df.groupby(['VendorID', 'VendorName', 'ItemName']).agg({
                'Quantity': 'sum',
                'Rate': 'last'
            }).reset_index()
            
            recommendations = []
            
            for _, item in inventory.iterrows():
                try:
                    current_stock = float(item['Quantity'])
                    prediction = self.predict_stock_level(
                        str(item['VendorID']),
                        str(item['ItemName']),
                        current_stock
                    )
                    
                    # Always add to recommendations
                    try:
                        rate = float(str(item['Rate']).replace(',', ''))
                    except (ValueError, TypeError):
                        rate = 0.0
                    
                    recommendations.append({
                        "vendor_id": str(item['VendorID']),
                        "vendor_name": str(item['VendorName']),
                        "item_name": str(item['ItemName']),
                        "current_stock": float(current_stock),
                        "predicted_stock": float(prediction['predicted_stock']),
                        "recommended_quantity": float(prediction['recommended_order']),
                        "rate": rate,
                        "confidence_score": prediction['confidence_score']
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing item {item['ItemName']}: {str(e)}")
                    continue
            
            # Sort by urgency and price
            recommendations.sort(key=lambda x: (x['predicted_stock'], x['rate']))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting ML recommendations: {str(e)}")
            raise

    async def check_and_generate_pos(self):
        """Check inventory using ML predictions and generate POs"""
        try:
            df = load_data()
            if df.empty:
                raise ValueError("No data found in the Excel file")
            
            # Get current inventory levels
            inventory = df.groupby(['VendorID', 'VendorName', 'ItemName']).agg({
                'Quantity': 'sum',
                'BillDate': 'max'
            }).reset_index()
            
            pos_generated = []
            
            for _, item in inventory.iterrows():
                current_stock = float(item['Quantity'])
                
                # Use ML model to predict future stock levels
                prediction = self.predict_stock_level(
                    item['VendorID'],
                    item['ItemName'],
                    current_stock
                )
                
                # Check if reorder is needed based on ML prediction
                if prediction['needs_reorder']:
                    try:
                        reorder_quantity = prediction['recommended_order']
                        
                        # Prepare PO items with ItemName key
                        po_items = [{
                            "ItemName": item['ItemName'],  # Using ItemName consistently
                            "quantity": float(reorder_quantity)
                        }]
                        
                        # Generate PO
                        po = await generate_purchase_order(
                            vendor_id=item['VendorID'],
                            items=po_items
                        )
                        
                        if po:
                            pos_generated.append({
                                "vendor_name": item['VendorName'],
                                "item_name": item['ItemName'],
                                "current_stock": round(current_stock, 2),
                                "predicted_stock": round(prediction['predicted_stock'], 2),
                                "recommended_quantity": round(reorder_quantity, 2),
                                "po_number": po.po_number,
                                "total_amount": round(po.total_amount, 2)
                            })
                            
                    except Exception as e:
                        print(f"Error generating PO for {item['ItemName']}: {str(e)}")
                        continue
            
            return pos_generated
            
        except Exception as e:
            print(f"Error in automatic PO generation: {str(e)}")
            raise

# Create global instance
predictor = POPredictor()

# Data models
class ItemPrediction(BaseModel):
    item_name: str
    predicted_quantity: float
    confidence_score: Optional[float] = None

class OrderItem(BaseModel):
    BillNumber: str
    BillDate: str
    DueDate: str
    VendorID: str
    VendorName: str
    CustomerID: str = ""
    CustomerName: str = ""
    ItemName: str
    Quantity: float
    Rate: float
    ItemTotal: float
    Source: str = "Client"
    PaymentTerms: str = "14"
    CurrencySymbol: str = "ZMW"

class PurchaseOrder(BaseModel):
    po_number: str
    created_date: str
    total_amount: float
    items: List[OrderItem]
    status: str = "created"

class POItem(BaseModel):
    """Schema for Purchase Order Item"""
    ItemName: str = Field(..., description="Name of the item")
    quantity: float = Field(..., description="Quantity to order")

class PORequest(BaseModel):
    """Schema for Purchase Order Request"""
    vendor_id: str = Field(..., description="Vendor ID")
    items: List[POItem] = Field(..., description="List of items to order")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vendor_id": "RRM219",
                "items": [
                    {
                        "ItemName": "Beef Fillet",
                        "quantity": 10.0
                    }
                ]
            }
        }

class PurchaseOrderResponse(BaseModel):
    po_number: str
    created_date: str
    total_amount: float
    items: List[dict]
    status: str

class DataManager:
    def __init__(self):
        self.df = None
        self.vendors = None
        self.customers = None
        self.items = None
        self.load_data()

    def load_data(self):
        """Load and cache data from Excel file"""
        try:
            self.df = pd.read_excel(PO_DATA_FILE, sheet_name='Orders')
            self.vendors = self.df[['VendorID', 'VendorName']].drop_duplicates()
            self.customers = self.df[['CustomerID', 'CustomerName']].drop_duplicates()
            self.items = self.df[['ItemName', 'Rate', 'VendorID']].drop_duplicates()
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def get_random_vendor(self) -> dict:
        """Get a random vendor from dataset"""
        vendor = self.vendors.sample(n=1).iloc[0]
        return {
            'VendorID': vendor['VendorID'],
            'VendorName': vendor['VendorName']
        }

    def get_random_customer(self) -> dict:
        """Get a random customer from dataset"""
        customer = self.customers.sample(n=1).iloc[0]
        return {
            'CustomerID': customer['CustomerID'],
            'CustomerName': customer['CustomerName']
        }

    def get_vendor_items(self, vendor_id: str, num_items: int = None) -> list:
        """Get random items for a vendor with actual rates"""
        try:
            vendor_items = self.df[self.df['VendorID'] == vendor_id].drop_duplicates(subset=['ItemName'])
            
            if vendor_items.empty:
                raise ValueError(f"No items found for vendor {vendor_id}")
            
            if num_items is None:
                num_items = np.random.randint(1, min(4, len(vendor_items) + 1))
                
            selected_items = []
            sampled_items = vendor_items.sample(n=min(num_items, len(vendor_items)))
            
            for _, item in sampled_items.iterrows():
                selected_items.append({
                    'ItemName': item['ItemName'],
                    'Rate': float(str(item['Rate']).replace(',', '')),
                    'Quantity': float(item['Quantity']),
                    'ItemTotal': float(str(item['ItemTotal']).replace(',', ''))
                })
                
            return selected_items
            
        except Exception as e:
            logger.error(f"Error getting vendor items: {str(e)}")
            raise

    def get_vendor_bill_number(self, vendor_id: str) -> str:
        """Get a bill number specific to vendor"""
        try:
            vendor_bills = self.df[self.df['VendorID'] == vendor_id]['BillNumber'].unique()
            if len(vendor_bills) > 0:
                return np.random.choice(vendor_bills)
            return "PO-001"
        except Exception as e:
            logger.error(f"Error getting vendor bill number: {str(e)}")
            return "PO-001"

# Create global instance
data_manager = DataManager()

# Helper functions
def load_data():
    """Load and validate data from Excel file"""
    try:
        return pd.read_excel(PO_DATA_FILE, sheet_name='Orders')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

def get_bill_number_from_dataset() -> str:
    """Get an existing bill number from the dataset"""
    try:
        df = pd.read_excel(PO_DATA_FILE, sheet_name='Orders')
        
        # Get a bill number for the specific vendor-customer combination
        bill_numbers = df['BillNumber'].unique()
        
        # Print available bill numbers for debugging
        print("Available bill numbers in dataset:", bill_numbers)
        
        # Use the first bill number from dataset
        return bill_numbers[0] if len(bill_numbers) > 0 else "PO-001"
        
    except Exception as e:
        print(f"Error getting bill number: {str(e)}")
        return "PO-001"

def calculate_due_date(created_date: datetime, payment_terms: int = 14):
    """Calculate due date based on payment terms"""
    return created_date + timedelta(days=payment_terms)

def extract_rate(rate_str):
    """Helper function to convert rate string or number to float"""
    try:
        # If already a number, just convert to float
        if isinstance(rate_str, (int, float, np.integer, np.floating)):
            return float(rate_str)
        # If string, remove commas and convert to float
        return float(rate_str.replace(',', '').strip())
    except (ValueError, AttributeError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid rate format: {rate_str}")

@app.on_event("startup")
async def startup_event():
    """Initialize and train ML model on startup"""
    try:
        df = load_data()
        predictor.train(df)
    except Exception as e:
        print(f"Error training ML model: {str(e)}")

class BillNumberManager:
    def __init__(self):
        self.bill_numbers = []
        self.bill_cycle = None
        self.current_index = 0
        self.load_bill_numbers()
    
    def load_bill_numbers(self):
        """Load bill numbers from dataset"""
        try:
            df = pd.read_excel(PO_DATA_FILE, sheet_name='Orders')
            self.bill_numbers = df['BillNumber'].unique().tolist()
            self.bill_cycle = cycle(self.bill_numbers)
            print(f"Loaded {len(self.bill_numbers)} bill numbers from dataset")
        except Exception as e:
            print(f"Error loading bill numbers: {str(e)}")
            self.bill_numbers = ["PO-001"]
            self.bill_cycle = cycle(self.bill_numbers)
    
    def get_next_bill_number(self) -> str:
        """Get next bill number in rotation"""
        if not self.bill_cycle:
            self.load_bill_numbers()
        return next(self.bill_cycle)

# Create a global instance
bill_manager = BillNumberManager()

def get_vendor_specific_bill_number(vendor_id: str, df: pd.DataFrame) -> str:
    """Get bill number specific to vendor from dataset"""
    try:
        # Filter for this vendor's bills
        vendor_bills = df[df['VendorID'] == vendor_id]['BillNumber'].unique()
        if len(vendor_bills) > 0:
            # Get a random bill number for this vendor
            return np.random.choice(vendor_bills)
        return "PO-001"
    except Exception as e:
        print(f"Error getting vendor bill number: {str(e)}")
        return "PO-001"

def get_random_vendor(df: pd.DataFrame) -> dict:
    """Get a random vendor from the dataset"""
    vendors = df[['VendorID', 'VendorName']].drop_duplicates()
    vendor = vendors.sample(n=1).iloc[0]
    return {
        'VendorID': vendor['VendorID'],
        'VendorName': vendor['VendorName']
    }

def get_random_customer(df: pd.DataFrame) -> dict:
    """Get a random customer from the dataset"""
    customers = df[['CustomerID', 'CustomerName']].drop_duplicates()
    customer = customers.sample(n=1).iloc[0]
    return {
        'CustomerID': customer['CustomerID'],
        'CustomerName': customer['CustomerName']
    }

def get_random_vendor_item(vendor_id: str, df: pd.DataFrame) -> dict:
    """Get a random item from vendor's available items in dataset"""
    try:
        # Get all items for this vendor
        vendor_items = df[df['VendorID'] == vendor_id][
            ['ItemName', 'Rate', 'Quantity', 'ItemTotal']
        ].drop_duplicates()
        
        if vendor_items.empty:
            raise ValueError(f"No items found for vendor {vendor_id}")
            
        # Select a random item
        item = vendor_items.sample(n=1).iloc[0]
        
        return {
            'ItemName': item['ItemName'],
            'Rate': float(str(item['Rate']).replace(',', '')),
            'Quantity': float(item['Quantity']),
            'ItemTotal': float(str(item['ItemTotal']).replace(',', ''))
        }
    except Exception as e:
        print(f"Error getting vendor item: {str(e)}")
        raise

@app.post("/generate-po", response_model=Union[PurchaseOrder, Dict[str, Any]])
async def generate_purchase_order(
    request: Optional[PORequest] = None
):
    """Generate purchase orders using ML predictions"""
    try:
        # Load all data
        df = load_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="No data found in database")

        po_items = []
        total_amount = 0
        po_number = str(np.random.randint(1000, 9999))
        today = datetime.now().strftime("%Y-%m-%d")
        due_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        # Define categories and related items
        meat_categories = {
            'beef': ['beef', 'steak', 'fillet', 'sirloin', 'rump', 'ribeye'],
            'chicken': ['chicken', 'wings', 'breast', 'thigh', 'drumstick'],
            'pork': ['pork', 'bacon', 'ham', 'sausage'],
            'other_meat': ['lamb', 'goat', 'turkey']
        }
        
        produce_categories = {
            'vegetables': ['tomato', 'potato', 'onion', 'carrot', 'cabbage'],
            'fruits': ['apple', 'banana', 'orange', 'grape', 'lemon']
        }

        # Get all vendors
        vendors = df['VendorID'].unique()
        
        if request and request.items:
            # For each requested item, find similar items from different vendors
            for requested_item in request.items:
                requested_name = requested_item.ItemName.lower()
                
                # Determine category of requested item
                category_items = []
                for cat, keywords in {**meat_categories, **produce_categories}.items():
                    if any(keyword in requested_name for keyword in keywords):
                        category_items = keywords
                        break
                
                # Find matching items from different vendors
                vendor_items = []
                for vendor_id in vendors:
                    vendor_df = df[df['VendorID'] == vendor_id]
                    
                    # Find best matching item from this vendor
                    best_match = None
                    best_similarity = 0
                    
                    for _, row in vendor_df.iterrows():
                        item_name = str(row['ItemName']).lower()
                        
                        # Check if item belongs to same category
                        if category_items and any(keyword in item_name for keyword in category_items):
                            similarity = fuzz.ratio(requested_name, item_name)
                            if similarity > best_similarity:
                                try:
                                    rate = float(str(row['Rate']).replace(',', ''))
                                    best_match = {
                                        'vendor_id': vendor_id,
                                        'vendor_name': row['VendorName'],
                                        'item_name': row['ItemName'],
                                        'rate': rate,
                                        'similarity': similarity
                                    }
                                    best_similarity = similarity
                                except (ValueError, TypeError):
                                    continue
                    
                    if best_match:
                        vendor_items.append(best_match)
                
                # Sort by price and take top 3 different vendors
                vendor_items.sort(key=lambda x: x['rate'])
                for vendor_item in vendor_items[:3]:
                    quantity = float(requested_item.quantity)
                    rate = vendor_item['rate']
                    item_total = quantity * rate
                    total_amount += item_total
                    
                    po_items.append(OrderItem(
                        BillNumber=po_number,
                        BillDate=today,
                        DueDate=due_date,
                        VendorID=vendor_item['vendor_id'],
                        VendorName=vendor_item['vendor_name'],
                        ItemName=vendor_item['item_name'],
                        Quantity=quantity,
                        Rate=rate,
                        ItemTotal=item_total
                    ))
                
                # Add complementary items
                if 'beef' in requested_name:
                    # Add vegetables with beef
                    for keyword in ['potato', 'onion', 'tomato']:
                        for vendor_id in vendors:
                            vendor_df = df[df['VendorID'] == vendor_id]
                            matching_items = vendor_df[vendor_df['ItemName'].str.lower().str.contains(keyword)]
                            
                            if not matching_items.empty:
                                item = matching_items.iloc[0]
                                try:
                                    rate = float(str(item['Rate']).replace(',', ''))
                                    quantity = 5.0  # Smaller quantity for complementary items
                                    item_total = quantity * rate
                                    total_amount += item_total
                                    
                                    po_items.append(OrderItem(
                                        BillNumber=po_number,
                                        BillDate=today,
                                        DueDate=due_date,
                                        VendorID=vendor_id,
                                        VendorName=item['VendorName'],
                                        ItemName=item['ItemName'],
                                        Quantity=quantity,
                                        Rate=rate,
                                        ItemTotal=item_total
                                    ))
                                    break
                                except (ValueError, TypeError):
                                    continue

        else:
            # Get a mix of items from different categories and vendors
            for vendor_id in vendors:
                vendor_df = df[df['VendorID'] == vendor_id]
                
                # Try to get one item from each category
                for category_dict in [meat_categories, produce_categories]:
                    for keywords in category_dict.values():
                        matching_items = vendor_df[
                            vendor_df['ItemName'].str.lower().apply(
                                lambda x: any(keyword in str(x).lower() for keyword in keywords)
                            )
                        ]
                        
                        if not matching_items.empty:
                            try:
                                item = matching_items.nsmallest(1, 'Rate').iloc[0]
                                rate = float(str(item['Rate']).replace(',', ''))
                                quantity = 10.0
                                item_total = quantity * rate
                                total_amount += item_total
                                
                                po_items.append(OrderItem(
                                    BillNumber=po_number,
                                    BillDate=today,
                                    DueDate=due_date,
                                    VendorID=vendor_id,
                                    VendorName=item['VendorName'],
                                    ItemName=item['ItemName'],
                                    Quantity=quantity,
                                    Rate=rate,
                                    ItemTotal=item_total
                                ))
                            except (ValueError, TypeError):
                                continue

        # Remove any duplicates while preserving vendor diversity
        unique_items = []
        seen = set()
        
        for item in po_items:
            key = (item.VendorID, item.ItemName)
            if key not in seen:
                unique_items.append(item)
                seen.add(key)

        # Update total amount
        total_amount = sum(item.ItemTotal for item in unique_items)

        if not unique_items:
            raise HTTPException(
                status_code=404,
                detail="No valid items found for purchase order"
            )

        return PurchaseOrder(
            po_number=po_number,
            created_date=today,
            total_amount=total_amount,
            items=unique_items,
            status="created"
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating PO: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating PO: {str(e)}"
        )

@app.get("/api/v1/check-stock-levels")
async def check_stock_levels():
    """Check stock levels and get reorder recommendations"""
    try:
        df = load_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="No data found in the Excel file")
        
        # Get unique vendors and their items
        vendors = df['VendorName'].unique()
        inventory_by_vendor = {}
        total_items = 0
        critical_count = 0
        low_stock_count = 0
        
        for vendor in vendors:
            # Filter data for this vendor
            vendor_data = df[df['VendorName'] == vendor]
            vendor_items = vendor_data.groupby('ItemName').agg({
                'Quantity': 'sum',
                'BillDate': 'last'
            }).reset_index()
            
            vendor_items_list = []
            
            for _, item in vendor_items.iterrows():
                current_quantity = float(item['Quantity'])
                
                # Determine stock status and recommended order
                if current_quantity <= 5:
                    recommended_order = max(10 - current_quantity, 5)
                    status = "CRITICAL LOW STOCK"
                    recommendation = "Immediate reorder required"
                    critical_count += 1
                elif current_quantity <= 10:
                    recommended_order = max(15 - current_quantity, 5)
                    status = "LOW STOCK"
                    recommendation = "Consider reordering soon"
                    low_stock_count += 1
                elif current_quantity <= 20:
                    recommended_order = max(25 - current_quantity, 5)
                    status = "MODERATE STOCK"
                    recommendation = "Monitor stock levels"
                else:
                    recommended_order = 0
                    status = "SUFFICIENT STOCK"
                    recommendation = "Stock levels are healthy"
                
                # Format the date properly
                try:
                    last_updated = pd.to_datetime(item['BillDate']).strftime('%Y-%m-%d') if pd.notnull(item['BillDate']) else None
                except:
                    last_updated = None
                
                vendor_items_list.append({
                    'item_name': item['ItemName'],
                    'quantity': {
                        'current': round(current_quantity, 2),
                        'recommended_order': round(recommended_order, 2),
                        'total_after_reorder': round(current_quantity + recommended_order, 2)
                    },
                    'status': status,
                    'recommendation': recommendation,
                    'last_updated': last_updated
                })
                
                total_items += 1
            
            # Sort vendor items by current quantity
            vendor_items_list.sort(key=lambda x: x['quantity']['current'])
            
            # Add vendor to inventory
            inventory_by_vendor[vendor] = {
                'items': vendor_items_list,
                'vendor_total': {
                    'total_items': len(vendor_items_list),
                    'critical_items': len([x for x in vendor_items_list if x['status'] == "CRITICAL LOW STOCK"]),
                    'low_stock_items': len([x for x in vendor_items_list if x['status'] == "LOW STOCK"])
                }
            }
        
        return {
            'status': 'success',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'inventory_by_vendor': inventory_by_vendor,
            'summary': {
                'total_items': total_items,
                'critical_items': critical_count,
                'low_stock_items': low_stock_count,
                'total_vendors': len(vendors)
            }
        }
        
    except Exception as e:
        print(f"Error in check_stock_levels: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional helper endpoints
@app.get("/api/v1/vendors")
async def get_vendors():
    """Get list of vendors with their details"""
    try:
        df = load_data()
        if df.empty:
            return {
                "status": "warning",
                "message": "No data found",
                "vendors": []
            }
        
        # First, get unique vendors with their IDs
        unique_vendors = df[['VendorName', 'VendorID']].drop_duplicates()
        
        vendors = []
        for _, vendor_row in unique_vendors.iterrows():
            vendors.append({
                "vendor_id": str(vendor_row['VendorID']),  # Ensure it's a string
                "vendor_name": str(vendor_row['VendorName'])  # Ensure it's a string
            })
        
        # Sort vendors by name
        vendors.sort(key=lambda x: x['vendor_name'])
        
        return {
            "status": "success",
            "count": len(vendors),
            "vendors": vendors
        }
        
    except Exception as e:
        print(f"Error in get_vendors: {str(e)}")  # Debug print
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching vendors: {str(e)}"
        )

@app.get("/api/v1/items")
async def get_items(
    category: ProductCategory = Query(
        default=ProductCategory.MEAT,
        description="Filter items by category"
    )
):
    """Get items with optional category filter"""
    try:
        # Get items from data manager
        items_df = data_manager.df[['ItemName', 'Rate', 'VendorName', 'VendorID']].drop_duplicates()
        
        if items_df.empty:
            return {
                "status": "warning",
                "message": "No items found in database",
                "data": []
            }
        
        # Process all items
        all_items = []
        for _, row in items_df.iterrows():
            try:
                rate = float(str(row['Rate']).replace(',', ''))
            except (ValueError, TypeError):
                rate = 0.0
                
            item_category = categorize_item(row['ItemName'])
            
            # Skip if category doesn't match (and not requesting all categories)
            if category != item_category:
                continue
                
            all_items.append({
                "item_name": row['ItemName'],
                "category": item_category,
                "price": {
                    "amount": rate,
                    "currency": "ZMW"  # Added currency
                },
                "vendor_name": row['VendorName'],
                "vendor_id": row['VendorID']
            })
        
        # Sort items by name
        all_items.sort(key=lambda x: x['item_name'])
        
        if not all_items:
            return {
                "status": "warning",
                "message": f"No items found for category: {category}",
                "data": []
            }
        
        return {
            "status": "success",
            "data": {
                "category": category,
                "currency": "ZMW",  # Added currency at top level
                "items": all_items,
                "total_items": len(all_items)
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching items: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching items: {str(e)}"
        )

@app.get("/api/v1/items/{vendor_name}")
async def get_items_by_vendor(vendor_name: str):
    """Get list of items filtered by vendor"""
    try:
        df = load_data()
        
        # Clean vendor name for case-insensitive comparison
        vendor_name = vendor_name.strip().lower()
        
        # Filter items for the specified vendor
        vendor_items = df[df['VendorName'].str.lower() == vendor_name][['ItemName', 'Rate', 'VendorName']].drop_duplicates()
        
        if vendor_items.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No items found for vendor: {vendor_name}"
            )
        
        items = []
        for _, item in vendor_items.iterrows():
            items.append({
                "item_name": item['ItemName'],
                "rate": float(str(item['Rate']).replace(',', '')),
                "vendor_name": item['VendorName']
            })
            
        return {
            "status": "success",
            "vendor": vendor_items['VendorName'].iloc[0],  # Use actual case from database
            "count": len(items),
            "items": items
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching items: {str(e)}"
        )

def categorize_item(item_name: str) -> str:
    """Categorize items based on actual dataset products"""
    item_name = item_name.lower()
    
    # Meat & Poultry
    if any(meat in item_name for meat in [
        'beef', 'chicken', 'pork', 'lamb', 'meat', 'sausage', 'bacon', 
        'mince', 'cubes', 'fillet', 'drumstick', 'wing', 'thigh',
        'breast', 'ox-tail', 'turkey', 'goat', 'rib', 'steak',
        'boerewors', 'chipolata', 'porker', 'baconer'
    ]):
        return ProductCategory.MEAT
    
    # Fruits & Vegetables
    elif any(produce in item_name for produce in [
        'apple', 'banana', 'orange', 'tomato', 'potato', 'onion',
        'cabbage', 'carrot', 'fruit', 'veg', 'vegetables', 'avocado',
        'lettuce', 'cucumber', 'pepper', 'garlic', 'butternut',
        'broccoli', 'cauliflower', 'celery', 'corn', 'eggplant',
        'ginger', 'grape', 'herb', 'lemon', 'lime', 'melon',
        'mint', 'mushroom', 'parsley', 'radish', 'rape',
        'spinach', 'strawberry', 'thyme', 'watermelon', 'zucchini',
        'chilli', 'beetroot', 'marrow', 'pakchoy', 'bean'
    ]):
        return ProductCategory.PRODUCE
    
    # Fish & Seafood
    elif any(seafood in item_name for seafood in [
        'fish', 'kapenta', 'bream', 'tilapia', 'seafood', 'prawns',
        'sardines', 'tuna', 'salmon'
    ]):
        return ProductCategory.SEAFOOD
    
    # Dairy & Eggs
    elif any(dairy in item_name for dairy in [
        'milk', 'cheese', 'yogurt', 'eggs', 'butter', 'cream',
        'dairy', 'margarine', 'yoghurt'
    ]):
        return ProductCategory.DAIRY
    
    # Default to MEAT if no category matches
    return ProductCategory.MEAT

class ItemResponse(BaseModel):
    item_name: str
    category: str
    rate: float
    vendor_name: str
    vendor_id: str

@app.get("/api/v1/auto-generate-pos")
async def auto_generate_purchase_orders():
    """Endpoint to trigger automatic PO generation for low stock items"""
    try:
        pos_generated = await predictor.check_and_generate_pos()
        
        if not pos_generated:
            return {
                "status": "success",
                "message": "No items require reordering at this time",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pos_generated": [],
                "total_pos": 0
            }
        
        return {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pos_generated": pos_generated,
            "total_pos": len(pos_generated)
        }
        
    except Exception as e:
        logger.error(f"Error generating automatic POs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating automatic POs: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # For local development
    uvicorn.run(
        "po_api:app",
        host=HOST,
        port=PORT,
        reload=True if os.getenv('ENVIRONMENT') == 'development' else False
    )