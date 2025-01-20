from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import signal
import sys
import time
from itertools import cycle

app = FastAPI(title="Purchase Order Creation API")

# Update path to use formatted dataset
BASE_PATH = r"C:\Users\NyashaKampila\Desktop\Projects\Purchase-Order-Creation-Project"
PO_DATA_FILE = os.path.join(BASE_PATH, "PO Data_formatted.xlsx")

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
    item_name: str
    predicted_quantity: float = Field(gt=0)

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

def get_random_vendor_items(vendor_id: str, df: pd.DataFrame, num_items: int = None) -> list:
    """Get random items from vendor's available items in dataset"""
    try:
        # Get all items for this vendor
        vendor_items = df[df['VendorID'] == vendor_id].drop_duplicates(subset=['ItemName'])
        
        if vendor_items.empty:
            raise ValueError(f"No items found for vendor {vendor_id}")
        
        # Determine number of items to select
        if num_items is None:
            num_items = np.random.randint(1, min(4, len(vendor_items) + 1))
            
        # Select random items
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
        print(f"Error getting vendor items: {str(e)}")
        raise

@app.post("/api/v1/create-purchase-order", response_model=PurchaseOrderResponse)
async def create_purchase_order(request: PurchaseOrderRequest):
    try:
        # Load DataFrame
        df = pd.read_excel(PO_DATA_FILE, sheet_name='Orders')
        
        # Get random vendor instead of using request
        valid_vendor = get_random_vendor(df)
        print(f"Selected vendor: {valid_vendor['VendorName']} ({valid_vendor['VendorID']})")
        
        # Get random customer
        valid_customer = get_random_customer(df)
        print(f"Selected customer: {valid_customer['CustomerName']} ({valid_customer['CustomerID']})")
        
        # Get vendor-specific bill number
        bill_number = get_vendor_specific_bill_number(valid_vendor['VendorID'], df)
        print(f"Using bill number: {bill_number}")
        
        # Get random items for this vendor
        vendor_items = get_random_vendor_items(valid_vendor['VendorID'], df)
        
        # Process items
        processed_items = []
        total_amount = 0
        
        for item_details in vendor_items:
            processed_item = {
                'BillNumber': bill_number,
                'BillDate': datetime.now().strftime('%Y-%m-%d'),
                'DueDate': calculate_due_date(datetime.now()).strftime('%Y-%m-%d'),
                'VendorID': valid_vendor['VendorID'],
                'VendorName': valid_vendor['VendorName'],
                'CustomerID': valid_customer['CustomerID'],
                'CustomerName': valid_customer['CustomerName'],
                'ItemName': item_details['ItemName'],
                'Quantity': item_details['Quantity'],
                'Rate': item_details['Rate'],
                'ItemTotal': item_details['ItemTotal'],
                'Source': 'Client',
                'PaymentTerms': '14',
                'CurrencySymbol': 'ZMW'
            }
            
            processed_items.append(processed_item)
            total_amount += item_details['ItemTotal']

        return PurchaseOrderResponse(
            po_number=bill_number,
            created_date=processed_items[0]['BillDate'],
            total_amount=total_amount,
            items=processed_items,
            status="created"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing purchase order: {str(e)}"
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
async def get_items():
    """Get list of items"""
    try:
        df = load_data()
        # Only get unique items with their names
        items = df[['ItemName']].drop_duplicates().to_dict('records')
        return {
            "status": "success",
            "count": len(items),
            "items": items
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=str(e)
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

@app.get("/api/v1/vendors/{vendor_id}/items")
async def get_items_by_vendor_id(vendor_id: str):
    """Get list of items filtered by vendor ID"""
    try:
        df = load_data()
        
        # Clean vendor ID and make case-insensitive comparison
        vendor_id = vendor_id.strip().upper()
        
        # Filter items for the specified vendor ID
        vendor_items = df[df['VendorID'].str.upper() == vendor_id][
            ['ItemName', 'Rate', 'VendorName', 'VendorID']
        ].drop_duplicates()
        
        if vendor_items.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No items found for vendor ID: {vendor_id}"
            )
        
        items = []
        for _, item in vendor_items.iterrows():
            items.append({
                "item_name": item['ItemName'],
                "rate": float(str(item['Rate']).replace(',', '')),
                "vendor_name": item['VendorName'],
                "vendor_id": item['VendorID']
            })
            
        return {
            "status": "success",
            "vendor": {
                "id": vendor_id,
                "name": vendor_items['VendorName'].iloc[0]
            },
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

@app.get("/api/v1/items/customer/{customer_name}")
async def get_items_by_customer(customer_name: str):
    """Get list of items filtered by customer"""
    try:
        df = load_data()
        
        # Clean customer name for case-insensitive comparison
        customer_name = customer_name.strip().lower()
        
        # Filter items for the specified customer
        customer_items = df[df['CustomerName'].str.lower() == customer_name][
            ['ItemName', 'Rate', 'CustomerName', 'VendorName']
        ].drop_duplicates()
        
        if customer_items.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No items found for customer: {customer_name}"
            )
        
        items = []
        for _, item in customer_items.iterrows():
            items.append({
                "item_name": item['ItemName'],
                "rate": float(str(item['Rate']).replace(',', '')),
                "customer_name": item['CustomerName'],
                "vendor_name": item['VendorName']
            })
            
        return {
            "status": "success",
            "customer": customer_items['CustomerName'].iloc[0],  # Use actual case from database
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

@app.get("/api/v1/customers/{customer_id}/items")
async def get_items_by_customer_id(customer_id: str):
    """Get list of items filtered by customer ID"""
    try:
        df = load_data()
        
        # Clean customer ID and make case-insensitive comparison
        customer_id = customer_id.strip().upper()
        
        # Filter items for the specified customer ID
        customer_items = df[df['CustomerID'].str.upper() == customer_id][
            ['ItemName', 'Rate', 'CustomerName', 'CustomerID', 'VendorName']
        ].drop_duplicates()
        
        if customer_items.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No items found for customer ID: {customer_id}"
            )
        
        items = []
        for _, item in customer_items.iterrows():
            items.append({
                "item_name": item['ItemName'],
                "rate": float(str(item['Rate']).replace(',', '')),
                "customer_name": item['CustomerName'],
                "customer_id": item['CustomerID'],
                "vendor_name": item['VendorName']
            })
            
        return {
            "status": "success",
            "customer": {
                "id": customer_id,
                "name": customer_items['CustomerName'].iloc[0]
            },
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

@app.get("/api/v1/available-bill-numbers")
async def get_available_bill_numbers():
    """Get list of available bill numbers"""
    return {
        "status": "success",
        "count": len(bill_manager.bill_numbers),
        "bill_numbers": bill_manager.bill_numbers
    }

# Add an endpoint to see available items for a vendor
@app.get("/api/v1/vendors/{vendor_id}/items")
async def get_vendor_available_items(vendor_id: str):
    """Get list of available items for a vendor"""
    try:
        df = pd.read_excel(PO_DATA_FILE, sheet_name='Orders')
        vendor_items = df[df['VendorID'] == vendor_id][
            ['ItemName', 'Rate', 'Quantity']
        ].drop_duplicates().to_dict('records')
        
        return {
            "status": "success",
            "vendor_id": vendor_id,
            "count": len(vendor_items),
            "items": vendor_items
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching vendor items: {str(e)}"
        )

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    import uvicorn
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1
    )
    
    server = uvicorn.Server(config)
    try:
        print("Starting server... Press Ctrl+C to quit")
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)