//
//  ViewController.swift
//  MyWeather2
//
//  Created by Hanqing Liu on 12/2/20.
//

import UIKit
import AVFoundation
import CoreLocation

// Global Variables
let userName = "Gregg"
let cellTextFontBold = UIFont(name: "SFCompactDisplay-Bold", size: 18)
let cellTextFontMedium = UIFont(name: "SFCompactDisplay-Medium", size: 18)
let cellTextWidth = 170
let cellTextSize = CGFloat(18)
let cellTextWeight = UIFont.Weight.semibold
let weatherSymbolFrame = [55, 53]

let backgroundColor = UIColor(red: 126/255, green: 116/255, blue: 116/255, alpha: 1)
let collectionBGColor = UIColor(red: 196/255, green: 182/255, blue: 182/255, alpha: 0.8)
let buttonColor = UIColor(red: 255/255, green: 221/255, blue: 147/255, alpha: 1)
let mutedButtonColor = UIColor(displayP3Red: 255/255, green: 229/255, blue: 204/255, alpha: 1)
let cornerRadius = 30


class ViewController: UIViewController {
    @IBAction func longPressCollection(_ sender: UILongPressGestureRecognizer) {
        if sender.state == .began {
            guard let targetIndexPath = collectionView.indexPathForItem(at: sender.location(in: collectionView))
            else {
                return
            }
            AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
            collectionView.beginInteractiveMovementForItem(at: targetIndexPath)
        } else if sender.state == .changed {
            collectionView.updateInteractiveMovementTargetPosition(sender.location(in: collectionView))
        } else if sender.state == .ended {
            collectionView.endInteractiveMovement()
        } else {
            collectionView.cancelInteractiveMovement()
        }
    }
    
    @IBOutlet weak var mainVertStack: UIStackView!
    @IBOutlet weak var collectionView: UICollectionView!
    @IBAction func readButton(_ sender: UIButton) {
        if self.addMode == false {
            self.collectionView.reloadData()
            for item in weatherObjectsToShow {
                if item.title == "Description" {
                    textToRead += "we have a \(item.value), "
                } else if item.title == "Temp." {
                    textToRead += "the temperature is \(item.value) Fahrenheit, "
                } else if item.title == "Pressure" {
                    textToRead += "the atmospheric pressure is \(item.value) hectopascals, "
                } else if item.title == "Humidity" {
                    textToRead += "the humidity is \(item.value) percent, "
                } else if item.title == "Wind Speed" {
                    textToRead += "the wind speed is \(item.value) miles per hour, "
                } else if item.title == "Wind Deg." {
                    textToRead += "the wind direction is \(item.value) degrees, "
                } else if item.title == "Cloudiness" {
                    textToRead += "the cloudiness is \(item.value) percent, "
                } else if item.title == "Visibility" {
                    textToRead += "the visibility is \(item.value) meters, "
                }
            }
            textToRead += "I wish you have a wonderful day!"
            let utterance = AVSpeechUtterance(string: textToRead)
            synth.speak(utterance)
        } else {
            // When Done is pushed
            showReadItButton()
            self.addMode = false
            var tempWeatherToShow = [WeatherItem]()
            for item in self.weatherObjects {
                if item.done == true {
                    tempWeatherToShow.append(item)
                }
            }
            self.weatherObjectsToShow = tempWeatherToShow
            self.collectionView.reloadData()
        }
    }
    @IBOutlet weak var readButton: UIButton!
    
    // MARK: - Weather Object Lists
    let city = "Jersey+City"
    var weatherObjects = [WeatherItem(title: "Description", done: true), WeatherItem(title: "Temp.", done: true, unit: "°F"), WeatherItem(title: "Pressure", done: true, unit: "hPa"), WeatherItem(title: "Humidity", unit: "%"), WeatherItem(title: "Wind Speed", unit: "mph"), WeatherItem(title: "Wind Deg.", unit: "°"), WeatherItem(title: "Cloudiness", unit: "%"), WeatherItem(title: "Visibility", unit: "m")]
    var weatherObjectsToShow = [WeatherItem]()
    
    var weatherRequestManager = WeatherRequestManager()
    let locationManager = CLLocationManager()
    
    var textToRead = "Good morning \(userName), today "
    let synth = AVSpeechSynthesizer()
    
    var addMode = false
    
    // MARK: - viewDidLoad()
    override func viewDidLoad() {
        super.viewDidLoad()
        // Delegate
        weatherRequestManager.delegate = self
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.requestLocation()
        
        view.backgroundColor = backgroundColor
        tabBarController?.tabBar.barTintColor = backgroundColor
        
        settingUpCollectionView()
        settingUpFlowLayout()
        settingUpBotton()
        
        navigationController?.navigationBar.barTintColor = backgroundColor
        
        showReadItButton()
        
        // Wait for weather to load
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            for item in self.weatherObjects {
                if item.done == true {
                    self.weatherObjectsToShow.append(item)
                }
            }
            self.collectionView.reloadData()
            self.readButton.isEnabled = true
        }
    }
    
    func showErrorAlertWhenNoWifi() {
        let alert = UIAlertController(title: "Can't connect to internet", message: "Please check your internet connection", preferredStyle: .alert)
        let action = UIAlertAction(title: "Ok", style: .default) { (action) in
            // once Add Item Button clicked
            self.collectionView.reloadData()
        }
        
        alert.addAction(action)
        
        present(alert, animated: true, completion: nil)
    }
    
    func settingUpFlowLayout() {
        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .vertical
        layout.itemSize = CGSize(width: view.frame.size.width / 2.3, height: view.frame.size.width / 2.3)
        layout.sectionInset = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
        collectionView.collectionViewLayout = layout
    }
    
    func settingUpCollectionView() {
        collectionView.delegate = self
        collectionView.dataSource = self
        collectionView.backgroundColor = collectionBGColor
        collectionView.layer.cornerRadius = CGFloat(cornerRadius)
        collectionView.register(UICollectionViewCell.self, forCellWithReuseIdentifier: "cell")
    }
    
    func settingUpBotton() {
        readButton.widthAnchor.constraint(equalToConstant: 300).isActive = true
        readButton.heightAnchor.constraint(equalToConstant: 80).isActive = true
        readButton.backgroundColor = buttonColor
        readButton.layer.cornerRadius = CGFloat(cornerRadius)
        readButton.isEnabled = false
    }
    
    func showReadItButton() {
        readButton.setImage(UIImage(systemName: "speaker.wave.2.circle"), for: .normal)
        readButton.tintColor = .darkGray
        readButton.imageView?.contentMode = .scaleAspectFit
        readButton.contentVerticalAlignment = .fill
        readButton.contentHorizontalAlignment = .fill
        readButton.imageEdgeInsets = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
    }
}


extension ViewController: UICollectionViewDelegate, UICollectionViewDataSource, UICollectionViewDelegateFlowLayout {
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        if addMode == true {
            return weatherObjects.count
        } else {
            return weatherObjectsToShow.count + 1
        }
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "cell", for: indexPath)
        // In addMode show everything
        if addMode == true {
            // Chosen cells
            if weatherObjects[indexPath.row].done == true {
                cell.backgroundColor = buttonColor
                cell.layer.cornerRadius = CGFloat(cornerRadius)
                
                for subview in cell.subviews {
                    subview.removeFromSuperview()
                }
                
                let label = UILabel()
                label.adjustsFontForContentSizeCategory = true
                label.numberOfLines = 0
                label.textColor = .black
                label.frame = CGRect(x: 10, y: -20, width: cellTextWidth, height: 200)
                
                let weatherSymbolAttachment = NSTextAttachment()
                weatherSymbolAttachment.image = UIImage(systemName: weatherObjects[indexPath.row].imgText)?.withRenderingMode(.alwaysTemplate)
                weatherSymbolAttachment.bounds = CGRect(x: 0, y: -20, width: weatherSymbolFrame[0], height: weatherSymbolFrame[1])
                let fullString = NSMutableAttributedString(attachment: weatherSymbolAttachment)
                fullString.append(NSAttributedString(string: weatherObjects[indexPath.row].title + "\n", attributes: [NSAttributedString.Key.font: cellTextFontBold!]))
                fullString.append(NSAttributedString(string: weatherObjects[indexPath.row].value + " " + weatherObjects[indexPath.row].unit, attributes: [NSAttributedString.Key.font: UIFont.systemFont(ofSize: 22, weight: UIFont.Weight.bold)]))
                label.attributedText = fullString
                
                label.translatesAutoresizingMaskIntoConstraints = false
                
                cell.addSubview(label)
            } else {
                // Not chosen cells
                cell.backgroundColor = mutedButtonColor
                cell.layer.cornerRadius = CGFloat(cornerRadius)
                
                for subview in cell.subviews {
                    subview.removeFromSuperview()
                }
                
                let label = UILabel()
                label.adjustsFontForContentSizeCategory = true
                label.numberOfLines = 0
                label.textColor = .gray
                label.frame = CGRect(x: 10, y: -20, width: cellTextWidth, height: 200)
                
                let weatherSymbolAttachment = NSTextAttachment()
                weatherSymbolAttachment.image = UIImage(systemName: weatherObjects[indexPath.row].imgText)?.withRenderingMode(.alwaysTemplate)
                weatherSymbolAttachment.bounds = CGRect(x: 0, y: -20, width: weatherSymbolFrame[0], height: weatherSymbolFrame[1])
                let fullString = NSMutableAttributedString(attachment: weatherSymbolAttachment)
                fullString.append(NSAttributedString(string: weatherObjects[indexPath.row].title + "\n", attributes: [NSAttributedString.Key.font: cellTextFontMedium!]))
                fullString.append(NSAttributedString(string: weatherObjects[indexPath.row].value + " " + weatherObjects[indexPath.row].unit, attributes: [NSAttributedString.Key.font: UIFont.systemFont(ofSize: 22, weight: UIFont.Weight.regular)]))
                label.attributedText = fullString
                
                label.translatesAutoresizingMaskIntoConstraints = false
                
                cell.addSubview(label)
            }
            return cell
        } else {
            // Not in addMode
            if (indexPath.row < weatherObjectsToShow.count) {
                cell.backgroundColor = buttonColor
                cell.layer.cornerRadius = CGFloat(cornerRadius)
                
                for subview in cell.subviews {
                    subview.removeFromSuperview()
                }
                
                let label = UILabel()
                label.adjustsFontForContentSizeCategory = true
                label.numberOfLines = 0
                label.textColor = .black
                label.frame = CGRect(x: 10, y: -20, width: cellTextWidth, height: 200)
                
                let weatherSymbolAttachment = NSTextAttachment()
                weatherSymbolAttachment.image = UIImage(systemName: weatherObjectsToShow[indexPath.row].imgText)?.withRenderingMode(.alwaysTemplate)
                weatherSymbolAttachment.bounds = CGRect(x: 0, y: -20, width: weatherSymbolFrame[0], height: weatherSymbolFrame[1])
                let fullString = NSMutableAttributedString(attachment: weatherSymbolAttachment)
                fullString.append(NSAttributedString(string: weatherObjectsToShow[indexPath.row].title + "\n", attributes: [NSAttributedString.Key.font: cellTextFontBold!]))
                fullString.append(NSAttributedString(string: weatherObjectsToShow[indexPath.row].value + " " + weatherObjectsToShow[indexPath.row].unit, attributes: [NSAttributedString.Key.font: UIFont.systemFont(ofSize: 22, weight: UIFont.Weight.bold)]))
                label.attributedText = fullString
                
                label.translatesAutoresizingMaskIntoConstraints = false
                
                cell.addSubview(label)
            } else {
                cell.backgroundColor = mutedButtonColor
                cell.layer.cornerRadius = CGFloat(cornerRadius)
                
                for subview in cell.subviews {
                    subview.removeFromSuperview()
                }
                
                let label = UILabel()
                label.adjustsFontForContentSizeCategory = true
                label.numberOfLines = 0
                label.textColor = .darkGray
                label.frame = CGRect(x: 43, y: -13, width: cellTextWidth, height: 200)
                
                let weatherSymbolAttachment = NSTextAttachment()
                weatherSymbolAttachment.image = UIImage(systemName: "plus.circle")?.withRenderingMode(.alwaysTemplate)
                // The plus sign don't touch
                weatherSymbolAttachment.bounds = CGRect(x: 0, y: 0, width: 80, height: 75)
                let fullString = NSAttributedString(attachment: weatherSymbolAttachment)
                label.attributedText = fullString
                label.translatesAutoresizingMaskIntoConstraints = false
                
                cell.addSubview(label)
            }
            return cell
        }
    }
    // MARK: - Add Clicked View
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        if self.addMode == false && indexPath.row == weatherObjectsToShow.count {
            readButton.setImage(UIImage(systemName: "checkmark.circle"), for: .normal)
            self.addMode = true
            self.collectionView.reloadData()
        } else if self.addMode == true {
            if weatherObjects[indexPath.row].done == true {
                weatherObjects[indexPath.row].done = false
                self.collectionView.reloadData()
            } else {
                weatherObjects[indexPath.row].done = true
                self.collectionView.reloadData()
            }
        }
    }
    
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        return CGSize(width: view.frame.size.width / 2.3, height: view.frame.size.width / 2.3)
    }
    
    func collectionView(_ collectionView: UICollectionView, canMoveItemAt indexPath: IndexPath) -> Bool {
        return true
    }
    
    func collectionView(_ collectionView: UICollectionView, moveItemAt sourceIndexPath: IndexPath, to destinationIndexPath: IndexPath) {
        let item = weatherObjectsToShow.remove(at: sourceIndexPath.row)
        weatherObjectsToShow.insert(item, at: destinationIndexPath.row)
    }
}


// MARK: - Weather Delegate
extension ViewController: WeatherRequestManagerDelegate {
    func didUpdateWeather(_ weatherRequestManager: WeatherRequestManager, weather: WeatherModel) {
        for (index, _) in self.weatherObjects.enumerated() {
            if self.weatherObjects[index].title == "Description" {
                self.weatherObjects[index].value = String(weather.description)
                self.weatherObjects[index].imgText = "globe"
            } else if self.weatherObjects[index].title == "Temp." {
                self.weatherObjects[index].value = "\t" + String(weather.temperature)
                self.weatherObjects[index].imgText = "thermometer.sun"
            } else if self.weatherObjects[index].title == "Pressure" {
                self.weatherObjects[index].value = "\t" + String(weather.pressure)
                self.weatherObjects[index].imgText = "barometer"
            } else if self.weatherObjects[index].title == "Humidity" {
                self.weatherObjects[index].value = "\t" + String(weather.humidity)
                self.weatherObjects[index].imgText = "drop.triangle"
            } else if self.weatherObjects[index].title == "Wind Speed" {
                self.weatherObjects[index].value = "\t" + String(weather.windSpeed)
                self.weatherObjects[index].imgText = "wind"
            } else if self.weatherObjects[index].title == "Wind Deg." {
                self.weatherObjects[index].value = "\t" + String(weather.WindDegree)
                self.weatherObjects[index].imgText = "location.circle"
            } else if self.weatherObjects[index].title == "Cloudiness" {
                self.weatherObjects[index].value = "\t" + String(weather.cloudiness)
                self.weatherObjects[index].imgText = "cloud"
            } else if self.weatherObjects[index].title == "Visibility" {
                self.weatherObjects[index].value = "\t" + String(weather.visibility)
                self.weatherObjects[index].imgText = "sparkles"
            }
        }
    }
    
    func didFailWithError(error: Error) {
        print(error)
    }
}

extension ViewController: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.last {
            locationManager.stopUpdatingLocation()
            let lon = location.coordinate.longitude
            let lat = location.coordinate.latitude
            weatherRequestManager.fetchWeather(latitude: lat, longitude: lon)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print(error)
        showErrorAlertWhenNoWifi()
    }
}
