import yarp
import time

class ActionInterface:
    def __init__(self, out_port_name, format={}):
        yarp.Network.init()

        in_port_name = f"/metacub_dashboard/{out_port_name.split('/')[-1][:-1]}i"
        self.format = format
        self.port = yarp.BufferedPortBottle()
        self.port.open(in_port_name)
        while not yarp.Network.connect(out_port_name, in_port_name): 
            time.sleep(0.1)

    def read(self):
        bottle = self.port.read(False)  # non-blocking
        if not bottle or bottle.isNull():
            return None
        
        value = self.cast_bottle(bottle, self.format)
        return value

    def cast_bottle(self, bottle, format, name=''):
        if isinstance(format, dict):
            return {key: self.cast_bottle(bottle.find(key), format[key], key) for key in format}
        elif isinstance(format, list):
            return [self.cast_bottle(bottle.asList().get(i), format[i]) for i in range(len(format))]
        elif isinstance(format, str):
            if format == 'int':
                return bottle.asInt()
            elif format == 'float':
                return bottle.asFloat64()
            elif format == 'string':
                return bottle.asString()
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            raise ValueError(f"Unsupported format: {format}")