from airSigning import airSigning
from handDetector import handDetector

if __name__ == "__main__":

    handComponent = handDetector(maxHands=1)
    signComponent = airSigning(handComponent)

    signComponent.drawSign()
