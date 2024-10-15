from airSigning import AirSigning
from handDetector import handDetector

if __name__ == "__main__":

    handComponent = handDetector(maxHands=1)
    signComponent = AirSigning(handComponent)

    signComponent.drawSign()
