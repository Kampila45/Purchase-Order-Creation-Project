from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
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
class InventoryPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.is_trained = False

    def prepare_data(self, df):
        # Prepare features for ML model
        features_df = df[['SKU', 'CustomerName', 'VendorName', 'Quantity']].copy()
        
        # Encode categorical variables
        for column in ['SKU', 'CustomerName', 'VendorName']:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            features_df[column] = self.label_encoders[column].fit_transform(features_df[column])
        
        return features_df

    def train(self, df):
        features_df = self.prepare_data(df)
        X = features_df[['SKU', 'CustomerName', 'VendorName']]
        y = features_df['Quantity']
        
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, sku, customer_name, vendor_name):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Encode input values
        sku_encoded = self.label_encoders['SKU'].transform([sku])[0]
        customer_encoded = self.label_encoders['CustomerName'].transform([customer_name])[0]
        vendor_encoded = self.label_encoders['VendorName'].transform([vendor_name])[0]
        
        # Make prediction
        prediction = self.model.predict([[sku_encoded, customer_encoded, vendor_encoded]])[0]
        return max(0, round(prediction, 2))  # Ensure non-negative quantity

# Initialize ML model
predictor = InventoryPredictor()

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

class PurchaseOrderRequest(BaseModel):
    vendor_id: Optional[str] = None
    vendor_name: Optional[str] = None
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    items: Optional[List[OrderItem]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "vendor_id": "RRM219",
                "vendor_name": "R & R Meats",
                "customer_id": "",
                "customer_name": "",
                "items": [
                    {
                        "item_name": "Beef (Cubes)",
                        "predicted_quantity": 10
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
        print("ML model trained successfully")
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

@app.post("/generate-po", response_model=PurchaseOrder)
async def generate_purchase_order():
    try:
        # Get random vendor
        vendor = data_manager.df[['VendorID', 'VendorName']].drop_duplicates().sample(n=1).iloc[0]
        
        # Get items for this vendor
        vendor_items = data_manager.df[
            data_manager.df['VendorID'] == vendor['VendorID']
        ][['ItemName', 'Rate']].drop_duplicates()
        
        # Select 1-3 random items
        num_items = np.random.randint(1, 4)
        selected_items = vendor_items.sample(n=min(num_items, len(vendor_items)))
        
        # Generate random quantities between 1 and 20
        items = []
        total_amount = 0
        po_number = str(np.random.randint(1000, 9999))
        today = datetime.now().strftime("%Y-%m-%d")
        due_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        
        for _, item in selected_items.iterrows():
            quantity = np.random.randint(1, 21)  # Random quantity 1-20
            rate = float(item['Rate'])
            item_total = quantity * rate
            total_amount += item_total
            
            items.append(OrderItem(
                BillNumber=po_number,
                BillDate=today,
                DueDate=due_date,
                VendorID=vendor['VendorID'],
                VendorName=vendor['VendorName'],
                ItemName=item['ItemName'],
                Quantity=quantity,
                Rate=rate,
                ItemTotal=item_total
            ))
        
        return PurchaseOrder(
            po_number=po_number,
            created_date=today,
            total_amount=total_amount,
            items=items
        )
        
    except Exception as e:
        logger.error(f"Error generating purchase order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        total_inventory_value = 0
        
        for vendor in vendors:
            # Filter data for this vendor
            vendor_data = df[df['VendorName'] == vendor]
            vendor_items = vendor_data.groupby('ItemName').agg({
                'Quantity': 'sum',
                'Rate': 'last',
                'BillDate': 'last'
            }).reset_index()
            
            vendor_items_list = []
            vendor_total_value = 0
            vendor_reorder_value = 0
            
            for _, item in vendor_items.iterrows():
                current_quantity = float(item['Quantity'])
                unit_rate = round(float(item['Rate']), 2)
                
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
                
                # Calculate values
                current_value = round(current_quantity * unit_rate, 2)
                reorder_value = round(recommended_order * unit_rate, 2)
                
                vendor_total_value += current_value
                vendor_reorder_value += reorder_value
                total_inventory_value += current_value
                
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
                    'pricing': {
                        'unit_rate': unit_rate,
                        'current_value': current_value,
                        'reorder_cost': reorder_value,
                        'currency': 'ZMW'
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
                    'current_stock_value': round(vendor_total_value, 2),
                    'recommended_reorder_value': round(vendor_reorder_value, 2),
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
                'total_vendors': len(vendors),
                'total_inventory_value': round(total_inventory_value, 2),
                'currency': 'ZMW'
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
            # Filter data for this vendor
            vendor_data = df[df['VendorName'] == vendor_row['VendorName']]
            
            # Calculate vendor statistics
            total_items = vendor_data['ItemName'].nunique()
            total_quantity = vendor_data['Quantity'].sum()
            last_order = vendor_data['BillDate'].max()
            
            try:
                last_order_date = pd.to_datetime(last_order).strftime('%Y-%m-%d') if pd.notnull(last_order) else None
            except:
                last_order_date = None
            
            vendors.append({
                "vendor_id": str(vendor_row['VendorID']),  # Ensure it's a string
                "vendor_name": str(vendor_row['VendorName']),  # Ensure it's a string
                "total_items": int(total_items),
                "total_quantity": round(float(total_quantity), 2),
                "last_order_date": last_order_date
            })
        
        # Sort vendors by name
        vendors.sort(key=lambda x: x['vendor_name'])
        
        # Print debug information
        print(f"Found {len(vendors)} vendors")
        for vendor in vendors:
            print(f"Vendor: {vendor['vendor_name']}, ID: {vendor['vendor_id']}")
        
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
            
            # Skip if specific category requested and doesn't match
            if category != ProductCategory.MEAT and item_category != category:
                continue
                
            all_items.append({
                "item_name": row['ItemName'],
                "category": item_category,
                "rate": rate,
                "vendor_name": row['VendorName'],
                "vendor_id": row['VendorID']
            })
        
        # Sort items by category and name
        all_items.sort(key=lambda x: (x['category'], x['item_name']))
        
        if category != ProductCategory.MEAT:
            # Single category response
            if not all_items:
                return {
                    "status": "warning",
                    "message": f"No items found for category: {category}",
                    "data": {
                        "category": category,
                        "items": [],
                        "statistics": {
                            "total_items": 0,
                            "average_price": 0,
                            "price_range": {"min": 0, "max": 0}
                        }
                    }
                }
            
            total_value = sum(item['rate'] for item in all_items)
            avg_price = total_value / len(all_items)
            
            return {
                "status": "success",
                "data": {
                    "category": category,
                    "items": all_items,
                    "statistics": {
                        "total_items": len(all_items),
                        "average_price": round(avg_price, 2),
                        "price_range": {
                            "min": round(min(item['rate'] for item in all_items), 2),
                            "max": round(max(item['rate'] for item in all_items), 2)
                        }
                    }
                }
            }
        else:
            # Group items by category
            categories = {}
            for item in all_items:
                cat = item['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)
            
            # Calculate statistics for each category
            category_stats = []
            for cat, items in categories.items():
                total_value = sum(item['rate'] for item in items)
                avg_price = total_value / len(items)
                
                category_stats.append({
                    "category": cat,
                    "items": items,
                    "statistics": {
                        "total_items": len(items),
                        "average_price": round(avg_price, 2),
                        "price_range": {
                            "min": round(min(item['rate'] for item in items), 2),
                            "max": round(max(item['rate'] for item in items), 2)
                        }
                    }
                })
            
            return {
                "status": "success",
                "data": {
                    "categories": category_stats,
                    "summary": {
                        "total_categories": len(category_stats),
                        "total_products": len(all_items),
                        "categories_distribution": {
                            cat["category"]: cat["statistics"]["total_items"] 
                            for cat in category_stats
                        }
                    }
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
    
    # Fruits & Vegetables
    if any(produce in item_name for produce in [
        'apple', 'banana', 'orange', 'tomato', 'potato', 'onion',
        'cabbage', 'carrot', 'fruit', 'veg', 'vegetables',
        'lettuce', 'cucumber', 'pepper', 'garlic', 'butternut'
    ]):
        return ProductCategory.PRODUCE
    
    # Meat & Poultry
    elif any(meat in item_name for meat in [
        'beef', 'chicken', 'pork', 'lamb', 'meat', 'sausage', 'bacon', 
        'mince', 'cubes', 'fillet', 'drumstick', 'wing'
    ]):
        return ProductCategory.MEAT
    
    # Fish & Seafood
    elif any(seafood in item_name for seafood in [
        'fish', 'kapenta', 'bream', 'tilapia', 'seafood', 'prawns',
        'sardines', 'tuna'
    ]):
        return ProductCategory.SEAFOOD
    
    # Dairy & Eggs
    elif any(dairy in item_name for dairy in [
        'milk', 'cheese', 'yogurt', 'eggs', 'butter', 'cream',
        'dairy', 'margarine', 'yoghurt'
    ]):
        return ProductCategory.DAIRY
    
    # Return the first category as default if no match is found
    return ProductCategory.MEAT

class ItemResponse(BaseModel):
    item_name: str
    category: str
    rate: float
    vendor_name: str
    vendor_id: str

if __name__ == "__main__":
    import uvicorn
    
    # For local development
    uvicorn.run(
        "po_api:app",
        host=HOST,
        port=PORT,
        reload=True if os.getenv('ENVIRONMENT') == 'development' else False
    )