from datetime import datetime
from pydantic import BaseModel
import json


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.__str__()
        return super().default(o)


class User(BaseModel):
    id: int
    name = "John Doe"
    signup_ts: datetime | None = None
    friends: list[int] = []


def test_to_json():
    external_data = {
        "id": "123",
        "signup_ts": "2019-06-01 12:22",
        "friends": [1, 2, "3"],
    }
    user = User(**external_data)

    dump_str = json.dumps(user.dict(), cls=EnhancedJSONEncoder)

    user2 = User(**json.loads(dump_str))

    assert user == user2
