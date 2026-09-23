import itertools
import math
import unittest

import numpy as np

from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.lateral import get_max_angle_vm
from opendbc.car.subaru.carcontroller import get_safety_CP
from opendbc.car.subaru.carstate import CarState
from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, CarControllerParams, SubaruFlags
from opendbc.car.vehicle_model import VehicleModel


class TestSubaruFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          assert len(fw) == fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}"


class TestSubaruNoEngageOutsideAngleBound(unittest.TestCase):
  # engaging with the wheel beyond the panda's lateral accel bound would be blocked frame after frame
  def _steer_request(self, ci, cc):
    while True:
      _, msgs = ci.CC.update(cc.as_reader(), ci.CS, 0)
      for addr, data, _ in msgs:
        if addr == 0x124:
          return (data[1] >> 4) & 1

  def test_waits_for_wheel_inside_bound(self):
    speed = 13.4
    ci = CarInterface(CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023))
    ci.update([])
    bound = get_max_angle_vm(speed - 1.0, ci.CC.VM, ci.CC.p)
    cc = structs.CarControl(latActive=True)
    cc.actuators.steeringAngleDeg = 0.0
    ci.CS.out = structs.CarState(vEgo=speed, vEgoRaw=speed, steeringAngleDeg=-(bound + 5))
    self.assertEqual([self._steer_request(ci, cc) for _ in range(3)], [0, 0, 0])
    self.assertAlmostEqual(ci.CC.apply_angle_last, -(bound + 5), places=3)
    ci.CS.out = structs.CarState(vEgo=speed, vEgoRaw=speed, steeringAngleDeg=-(bound - 5))
    self.assertEqual(self._steer_request(ci, cc), 0)
    self.assertEqual(self._steer_request(ci, cc), 1)
    self.assertLess(abs(ci.CC.apply_angle_last), bound)
    ci.CS.out = structs.CarState(vEgo=speed, vEgoRaw=speed, steeringAngleDeg=-(bound + 5))
    self.assertEqual(self._steer_request(ci, cc), 1)
    self.assertLessEqual(abs(ci.CC.apply_angle_last), bound)

  def test_engages_inside_bound_immediately(self):
    speed = 13.24
    ci = CarInterface(CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023))
    ci.update([])
    ci.CS.out = structs.CarState(vEgo=speed, vEgoRaw=speed, steeringAngleDeg=57.61)
    cc = structs.CarControl(latActive=False)
    self._steer_request(ci, cc)
    cc.latActive = True
    cc.actuators.steeringAngleDeg = 51.6
    self.assertEqual(self._steer_request(ci, cc), 1)


class TestSubaruAngleLimits(unittest.TestCase):
  def setUp(self):
    cp = get_safety_CP()
    self.limits = CarControllerParams(cp)
    self.vm = VehicleModel(cp)

  def test_low_speed_deadband(self):
    for speed, desired, expected in ((3.9, 2.49, 0.0), (3.9, 2.5, 2.5), (3.9, -2.49, 0.0), (3.9, -2.5, -2.5), (4.0, 0.5, 0.5)):
      with self.subTest(speed=speed, desired=desired):
        cp = CarInterface.get_non_essential_params(CAR.SUBARU_CROSSTREK_2025)
        ci = CarInterface(cp)
        ci.update([])
        ci.CS.out = structs.CarState(vEgo=speed, vEgoRaw=speed)
        cc = structs.CarControl(latActive=True)
        cc.actuators.steeringAngleDeg = desired
        actuators, _ = ci.CC.update(cc.as_reader(), ci.CS, 0)
        self.assertAlmostEqual(actuators.steeringAngleDeg, expected)

  def test_safety_model_is_conservative(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.LKAS_ANGLE:
        continue
      vm = VehicleModel(CarInterface.get_non_essential_params(platform))
      for speed in np.linspace(1, 60, 120):
        with self.subTest(platform=platform, speed=speed):
          angle = min(get_max_angle_vm(speed, self.vm, self.limits), CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX)
          accel = vm.calc_curvature(math.radians(angle), speed, 0) * speed ** 2
          self.assertLessEqual(accel, CarControllerParams.ANGLE_LIMITS.MAX_LATERAL_ACCEL + 1e-6)


class TestSubaruCruiseState(unittest.TestCase):
  def test_angle_cruise_uses_es_status(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.LKAS_ANGLE:
        continue
      cp = CarInterface.get_non_essential_params(platform)
      cs = CarState(cp)
      parsers = cs.get_can_parsers(cp)
      cruise_parser = parsers[Bus.alt if cp.flags & SubaruFlags.GLOBAL_GEN2 else Bus.cam]
      brake_parser = parsers[Bus.alt if cp.flags & SubaruFlags.GLOBAL_GEN2 else Bus.pt]
      for brake_pressed, status, brake_status in itertools.product((False, True), repeat=3):
        with self.subTest(platform=platform, brake=brake_pressed, status=status, es_brake=brake_status):
          brake_parser.vl["Brake_Status"]["Brake"] = brake_pressed
          cruise_parser.vl["ES_Status"]["Cruise_Activated"] = status
          cruise_parser.vl["ES_Brake"]["Cruise_Activated"] = brake_status
          self.assertEqual(cs.update(parsers).cruiseState.enabled, status)

  def test_hybrid_cruise_uses_es_brake(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.HYBRID:
        continue
      cp = CarInterface.get_non_essential_params(platform)
      cs = CarState(cp)
      parsers = cs.get_can_parsers(cp)
      cruise_parser = parsers[Bus.alt if cp.flags & SubaruFlags.GLOBAL_GEN2 else Bus.cam]
      for enabled in (False, True):
        with self.subTest(platform=platform, enabled=enabled):
          cruise_parser.vl["ES_Brake"]["Cruise_Activated"] = enabled
          self.assertEqual(cs.update(parsers).cruiseState.enabled, enabled)


class TestSubaruAvailability(unittest.TestCase):
  def test_angle_control_is_development_only(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.LKAS_ANGLE:
        continue
      for is_release in (False, True):
        with self.subTest(platform=platform, is_release=is_release):
          cp = CarInterface.get_params(platform, gen_empty_fingerprint(), [], False, is_release, False)
          self.assertEqual(cp.dashcamOnly, is_release or platform not in (CAR.SUBARU_CROSSTREK_2025, CAR.SUBARU_OUTBACK_2023))
