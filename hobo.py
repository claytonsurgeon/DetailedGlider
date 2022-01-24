

from datetime import datetime
import datetime

# example = "2022-01-02T00:00:00.763061656"
example = "2022-01-02T00:00:00.763061"

date = datetime.datetime.strptime(example, "%Y-%m-%dT%H:%M:%S.%f")

print(date.timestamp())