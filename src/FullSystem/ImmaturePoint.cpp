/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"
#include "FullSystem/ResidualProjections.h"


namespace dso
{
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib)
: u(u_), v(v_), host(host_), my_type(type), idepth(0), lastTraceStatus(IPS_UNINITIALIZED)
{

	gradH.setZero();

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

        Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]);



		color[idx] = ptc[0];
		if(!std::isfinite(color[idx])) {energyTH=NAN; return;}


		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();

		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}

	energyTH = patternNum*setting_outlierTH;
	energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

	idepth_GT=0;
	quality=10000;
}

ImmaturePoint::~ImmaturePoint()
{
}



/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{
	if(lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;

	float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

	Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
	Vec3f pt = pr + hostToFrame_Kt*idepth;
	float u = pt[0] / pt[2];
	float v = pt[1] / pt[2];

	if(!(u > 4 && v > 4 && u < wG[0]-5 && v < hG[0]-5))
	{
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}
	else
	{
		return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
	}
}





double ImmaturePoint::linearizeResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float &Hdd, float &bd,
		float idepth)
{
	if(tmpRes->state_state == ResState::OOB)
		{ tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	// check OOB due to scale angle change.

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_RTll = precalc->PRE_RTll;
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	//const float * const Il = tmpRes->target->I;

	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
				PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

		// depth derivatives.
		float dxInterp = hitColor[1]*HCalib->fxl();
		float dyInterp = hitColor[2]*HCalib->fyl();
		float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

		hw *= weights[idx]*weights[idx];

		Hdd += (hw*d_idepth)*d_idepth;
		bd += (hw*residual)*d_idepth;
	}


	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}



}
