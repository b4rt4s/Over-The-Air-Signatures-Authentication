from airSigning import AirSigning
from handDetector import handDetector

if __name__ == "__main__":

    handComponent = handDetector()
    signComponent = AirSigning(handComponent)

    signComponent.drawSign()
