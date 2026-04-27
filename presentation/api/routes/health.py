from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix = '/api', tags = ['Health'])

@router.get('/health')
def health():
    return {
        'status_code': 200,
        'timestamp': datetime.now().isoformat()
    }